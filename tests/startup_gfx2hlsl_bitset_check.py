"""Scalar 32-bit bit set and clear (s_bitset0_b32, s_bitset1_b32) in tools/startup/gfx2hlsl.py (Refs #2023).

  (no arguments)                       translation, evaluator, refusal and mutation controls, pure Python
  binary <llvm-mc>                     reassemble every SOP1 and draw fixture source and compare with recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel pairs with FXC ps_5_1 when given and with DXC
                                       ps_6_0, draw one pixel per case and compare the four exported words

Semantics (AMD RDNA 2 ISA section 12.3 SOP1 opcodes 0x1b and 0x1d; LLVM 15 SOPInstructions.td):
  s_bitset0_b32 D, S0   D.u[S0.u[4:0]] = 0; SCC is unchanged.
  s_bitset1_b32 D, S0   D.u[S0.u[4:0]] = 1; SCC is unchanged.
"""
import re
import struct
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import VS_SOURCE, VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr, words_text  # noqa: E402

MASK = 0xffffffff
OPS = ('s_bitset0_b32', 's_bitset1_b32')
SOP1_OPCODE = {name: op for op, (name, _) in D.SOP1_OPS.items()}
SCC_UNSET = 2
REG = 5
SRC_REG = 6

SPECIAL = [0, 1, 0x7fffffff, 0x80000000, MASK, 0xaaaaaaaa, 0x55555555, 0x12345678, 0xdeadbeef]
SHIFTS = list(range(32)) + [32, 33, 37, 63, 64, 0x80000005, MASK]
INLINE_TESTS = (0, 1, 31, 32, 37, 64, -1, -16)
LITERAL_TESTS = (0x80000005, 0x7fffffe0, 0xffffffc5)

# Verified SOP1 encodings (AMD RDNA 2 SOP1: [31:23]=0x17d, [22:16]=sdst, [15:8]=op, [7:0]=ssrc0).
ENCODINGS = (
    ('s_bitset0_b32 s5, s6', '061b85be'),
    ('s_bitset1_b32 s5, s6', '061d85be'),
    ('s_bitset0_b32 s5, s5', '051b85be'),
    ('s_bitset1_b32 s5, s5', '051d85be'),
    ('s_bitset0_b32 s5, 0', '801b85be'),
    ('s_bitset1_b32 s5, 37', 'a51d85be'),
    ('s_bitset0_b32 s5, -1', 'c11b85be'),
    ('s_bitset0_b32 s5, m0', '7c1b85be'),
    ('s_bitset1_b32 s5, 0x80000005', 'ff1d85be05000080'),
)


def encode_sop1(name, sdst, ssrc0, literal=None):
    word = 0xbe800000 | ((sdst & 0x7f) << 16) | (SOP1_OPCODE[name] << 8) | (ssrc0 & 0xff)
    return struct.pack('<I', word) + (b'' if literal is None else struct.pack('<I', literal & MASK))


def encode_sop2(op_code, sdst, ssrc0, ssrc1):
    return struct.pack('<I', 0x80000000 | ((op_code & 0x7f) << 23) | ((sdst & 0x7f) << 16)
                       | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))


def encode_sopc(op_code, ssrc0, ssrc1):
    return struct.pack('<I', 0xbf000000 | ((op_code & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))


def encode_vop1_mov_s(vdst, ssrc):
    return struct.pack('<I', 0x7e000000 | ((vdst & 0xff) << 17) | (0x01 << 9) | (ssrc & 0x1ff))


def encode_inline(val):
    if 0 <= val <= 64:
        return 128 + val
    if -16 <= val <= -1:
        return 192 - val
    return None


def ref_bitset(op, d, bit_idx):
    mask = 1 << (bit_idx & 31)
    return ((d | mask) if op == 's_bitset1_b32' else (d & ~mask)) & MASK


def decode_one(words):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    return decoded[0][2], decoded[0][3]


def make_probe(mod):
    class Probe(mod.Stage):
        """A Stage with no pipeline and with the uniformity state of a real one."""

        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.masks, self.varying_s = set(), set()
            self.scc_uniform = True
            self.exec_known = True

    return Probe


def run_lines(lines, env):
    env = dict(env, true=1, false=0)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


# ---- Draw fixtures (normal binary32 words only, avoiding denormals/NaNs through mrt0 float4) ---------------
# s_add_u32 sN, sN, 0 materializes the literal as a runtime SGPR (unfolded), keeping runtime lowering under test.
POST_DRAW_ASM = 'exp mrt0 v0, v1, v2, v3 done vm\ns_endpgm\n'
POST_DRAW_WORDS = bytes.fromhex('0f1800f800010203' '000081bf')


def build_draw_fixtures():
    # Draw 1: runtime s_bitset1_b32 (s4, s5), runtime s_bitset0_b32 (s6, s7), self-alias s_bitset0_b32 (s8, s8),
    # and SCC true preservation across self-alias s_bitset1_b32 (s10, s10).
    d1_ops = [
        ('s_mov_b32 s4, 0x3f800005', encode_sop1('s_mov_b32', 4, 255, 0x3f800005)),
        ('s_mov_b32 s5, 37', encode_sop1('s_mov_b32', 5, encode_inline(37))),
        ('s_add_u32 s4, s4, 0', encode_sop2(0x00, 4, 4, 128)),
        ('s_add_u32 s5, s5, 0', encode_sop2(0x00, 5, 5, 128)),
        ('s_bitset1_b32 s4, s5', encode_sop1('s_bitset1_b32', 4, 5)),
        ('s_mov_b32 s6, 0xc0a00001', encode_sop1('s_mov_b32', 6, 255, 0xc0a00001)),
        ('s_mov_b32 s7, 0x8000001f', encode_sop1('s_mov_b32', 7, 255, 0x8000001f)),
        ('s_add_u32 s6, s6, 0', encode_sop2(0x00, 6, 6, 128)),
        ('s_add_u32 s7, s7, 0', encode_sop2(0x00, 7, 7, 128)),
        ('s_bitset0_b32 s6, s7', encode_sop1('s_bitset0_b32', 6, 7)),
        ('s_mov_b32 s8, 0xbf80001f', encode_sop1('s_mov_b32', 8, 255, 0xbf80001f)),
        ('s_add_u32 s8, s8, 0', encode_sop2(0x00, 8, 8, 128)),
        ('s_bitset0_b32 s8, s8', encode_sop1('s_bitset0_b32', 8, 8)),
        ('s_mov_b32 s10, 0x41200003', encode_sop1('s_mov_b32', 10, 255, 0x41200003)),
        ('s_add_u32 s10, s10, 0', encode_sop2(0x00, 10, 10, 128)),
        ('s_mov_b32 s12, 0x42480001', encode_sop1('s_mov_b32', 12, 255, 0x42480001)),
        ('s_mov_b32 s13, 0x43480001', encode_sop1('s_mov_b32', 13, 255, 0x43480001)),
        ('s_cmp_eq_u32 s4, s4', encode_sopc(0x06, 4, 4)),
        ('s_bitset1_b32 s10, s10', encode_sop1('s_bitset1_b32', 10, 10)),
        ('s_cselect_b32 s11, s12, s13', encode_sop2(0x0a, 11, 12, 13)),
        ('v_mov_b32_e32 v0, s4', encode_vop1_mov_s(0, 4)),
        ('v_mov_b32_e32 v1, s6', encode_vop1_mov_s(1, 6)),
        ('v_mov_b32_e32 v2, s8', encode_vop1_mov_s(2, 8)),
        ('v_mov_b32_e32 v3, s11', encode_vop1_mov_s(3, 11)),
    ]
    # Draw 2: constant-folded bitset1/bitset0, runtime self-alias s_bitset1_b32 (s10, s10),
    # and SCC false preservation across a non-zero runtime s_bitset1_b32 (s14, 0).
    d2_ops = [
        ('s_mov_b32 s4, 0x3f800005', encode_sop1('s_mov_b32', 4, 255, 0x3f800005)),
        ('s_bitset1_b32 s4, 37', encode_sop1('s_bitset1_b32', 4, encode_inline(37))),
        ('s_mov_b32 s6, 0xc0a00001', encode_sop1('s_mov_b32', 6, 255, 0xc0a00001)),
        ('s_bitset0_b32 s6, -1', encode_sop1('s_bitset0_b32', 6, encode_inline(-1))),
        ('s_mov_b32 s10, 0x41200003', encode_sop1('s_mov_b32', 10, 255, 0x41200003)),
        ('s_add_u32 s10, s10, 0', encode_sop2(0x00, 10, 10, 128)),
        ('s_bitset1_b32 s10, s10', encode_sop1('s_bitset1_b32', 10, 10)),
        ('s_mov_b32 s14, 0x3f800005', encode_sop1('s_mov_b32', 14, 255, 0x3f800005)),
        ('s_add_u32 s14, s14, 0', encode_sop2(0x00, 14, 14, 128)),
        ('s_mov_b32 s12, 0x42480001', encode_sop1('s_mov_b32', 12, 255, 0x42480001)),
        ('s_mov_b32 s13, 0x43480001', encode_sop1('s_mov_b32', 13, 255, 0x43480001)),
        ('s_cmp_lg_u32 s14, s14', encode_sopc(0x07, 14, 14)),
        ('s_bitset1_b32 s14, 0', encode_sop1('s_bitset1_b32', 14, encode_inline(0))),
        ('s_cselect_b32 s11, s12, s13', encode_sop2(0x0a, 11, 12, 13)),
        ('v_mov_b32_e32 v0, s4', encode_vop1_mov_s(0, 4)),
        ('v_mov_b32_e32 v1, s6', encode_vop1_mov_s(1, 6)),
        ('v_mov_b32_e32 v2, s10', encode_vop1_mov_s(2, 10)),
        ('v_mov_b32_e32 v3, s11', encode_vop1_mov_s(3, 11)),
    ]
    d1_expected = (
        ref_bitset('s_bitset1_b32', 0x3f800005, 37),
        ref_bitset('s_bitset0_b32', 0xc0a00001, 0x8000001f),
        ref_bitset('s_bitset0_b32', 0xbf80001f, 0xbf80001f),
        0x42480001,
    )
    d1_mutated = (
        ref_bitset('s_bitset0_b32', 0x3f800005, 37),
        ref_bitset('s_bitset1_b32', 0xc0a00001, 0x8000001f),
        ref_bitset('s_bitset1_b32', 0xbf80001f, 0xbf80001f),
        0x42480001,
    )
    d2_expected = (
        ref_bitset('s_bitset1_b32', 0x3f800005, 37),
        ref_bitset('s_bitset0_b32', 0xc0a00001, -1),
        ref_bitset('s_bitset1_b32', 0x41200003, 0x41200003),
        0x43480001,
    )
    return (
        ('runtime_and_scc_true', '\n'.join(a for a, _ in d1_ops) + '\n' + POST_DRAW_ASM,
         b''.join(w for _, w in d1_ops) + POST_DRAW_WORDS, d1_expected, d1_mutated),
        ('constfold_and_scc_false', '\n'.join(a for a, _ in d2_ops) + '\n' + POST_DRAW_ASM,
         b''.join(w for _, w in d2_ops) + POST_DRAW_WORDS, d2_expected, None),
    )


DRAW_FIXTURES = build_draw_fixtures()


def is_normal_f32(word):
    return 0 < ((word >> 23) & 0xff) < 0xff


def mutate_draw_hlsl(text):
    """Flip s_bitset1_b32 (| (1u <<) and s_bitset0_b32 (& ~(1u <<) in runtime s4/s6/s8 assignments."""
    lines = []
    changed = 0
    for line in text.splitlines():
        m = re.match(r'^(\s*s[468] = )(s[468] )(\| \(1u <<|& ~\(1u <<)(.+)$', line)
        if m:
            flipped = '& ~(1u <<' if m.group(3).startswith('|') else '| (1u <<'
            line = m.group(1) + m.group(2) + flipped + m.group(4)
            changed += 1
        lines.append(line)
    assert changed == 3, (changed, text)
    return '\n'.join(lines) + '\n'


def battery(mod):
    """Every check as (label, ok, detail) against the translator module `mod`."""
    results = []
    Probe = make_probe(mod)

    def check(label, ok, detail=''):
        results.append((label, bool(ok), detail))

    def translate(words, setup=None):
        text_name, asm = decode_one(words)
        stage = Probe()
        if setup:
            setup(stage)
        stage.one(text_name, asm)
        return asm, stage

    def refusal(asm_name, asm, setup=None):
        stage = Probe()
        if setup:
            setup(stage)
        before = (set(stage.masks), dict(stage.sym), set(stage.varying_s), stage.scc_uniform, set(stage.used_s))
        try:
            stage.one(asm_name, asm)
        except mod.Unsupported as error:
            after = (stage.masks, stage.sym, stage.varying_s, stage.scc_uniform, stage.used_s)
            return str(error), not stage.lines and after == before
        return None, False

    check('SOP1 encodings match recorded words and decode back',
          all((lambda n, a: encode_sop1(n, *args).hex() == hex_str and decode_one(encode_sop1(n, *args))[1] == asm)(
              asm.split()[0], asm)
              for asm, hex_str in ENCODINGS
              for args in [(
                  int(asm.split()[1].rstrip(',')[1:]),
                  255 if asm.split()[2].startswith('0x') else (
                      124 if asm.split()[2] == 'm0' else (
                          int(asm.split()[2][1:]) if asm.split()[2].startswith('s')
                          else encode_inline(int(asm.split()[2])))),
                  int(asm.split()[2], 16) if asm.split()[2].startswith('0x') else None,
              )]))

    # 1. Runtime evaluation against Python reference across SGPR, inline, literal, m0 and D==S0 aliasing.
    for op in OPS:
        bad = None
        # SGPR index s6 across SPECIAL x SHIFTS x 3 SCC inputs
        _, st_sgpr = translate(encode_sop1(op, REG, SRC_REG))
        for d_val in SPECIAL:
            for s_val in SHIFTS:
                for scc_in in (SCC_UNSET, 0, 1):
                    env = run_lines(st_sgpr.lines, {'s%d' % REG: d_val, 's%d' % SRC_REG: s_val, 'scc': scc_in})
                    want = ref_bitset(op, d_val, s_val)
                    if (env['s%d' % REG], env['scc']) != (want, scc_in):
                        bad = ('sgpr', hex(d_val), hex(s_val), scc_in, env)
                        break
                if bad:
                    break
            if bad:
                break
        if bad is None:
            # Inline constants
            for imm in INLINE_TESTS:
                _, st_imm = translate(encode_sop1(op, REG, encode_inline(imm)))
                for d_val in SPECIAL:
                    env = run_lines(st_imm.lines, {'s%d' % REG: d_val, 'scc': SCC_UNSET})
                    want = ref_bitset(op, d_val, imm)
                    if (env['s%d' % REG], env['scc']) != (want, SCC_UNSET):
                        bad = ('inline', imm, hex(d_val), env)
                        break
                if bad:
                    break
        if bad is None:
            # 32-bit literals
            for lit in LITERAL_TESTS:
                _, st_lit = translate(encode_sop1(op, REG, 255, lit))
                for d_val in SPECIAL:
                    env = run_lines(st_lit.lines, {'s%d' % REG: d_val, 'scc': SCC_UNSET})
                    want = ref_bitset(op, d_val, lit)
                    if (env['s%d' % REG], env['scc']) != (want, SCC_UNSET):
                        bad = ('literal', hex(lit), hex(d_val), env)
                        break
                if bad:
                    break
        if bad is None:
            # m0 source and D == S0 aliasing
            _, st_m0 = translate(encode_sop1(op, REG, 124))
            _, st_alias = translate(encode_sop1(op, REG, REG))
            for d_val in SPECIAL + [0x00000025, 0x8000001f, 0x00000003]:
                env_m0 = run_lines(st_m0.lines, {'s%d' % REG: d_val, 'm0': 37, 'scc': SCC_UNSET})
                if (env_m0['s%d' % REG], env_m0['scc']) != (ref_bitset(op, d_val, 37), SCC_UNSET):
                    bad = ('m0', hex(d_val), env_m0)
                    break
                env_al = run_lines(st_alias.lines, {'s%d' % REG: d_val, 'scc': SCC_UNSET})
                if (env_al['s%d' % REG], env_al['scc']) != (ref_bitset(op, d_val, d_val), SCC_UNSET):
                    bad = ('alias', hex(d_val), env_al)
                    break
        check('%s: SGPR/inline/literal/m0/self-alias match reference with SCC unchanged' % op, bad is None, str(bad))

    # 2. SCC preservation across s_cmp -> s_bitset1 -> s_bitset0 -> s_cselect.
    scc_seq_ok = True
    for cmp_op, want in (('s_cmp_eq_u32 s7, s7', 1), ('s_cmp_lg_u32 s7, s7', 0)):
        st = Probe()
        st.one(cmp_op.split()[0], cmp_op)
        st.one('s_bitset1_b32', 's_bitset1_b32 s%d, 0' % REG)
        st.one('s_bitset0_b32', 's_bitset0_b32 s%d, 31' % REG)
        st.one('s_cselect_b32', 's_cselect_b32 s9, 1, 0')
        env = run_lines(st.lines, {'s%d' % REG: 0x80000000, 's7': 3, 'scc': SCC_UNSET})
        if env['s9'] != want or env['s%d' % REG] != 1:
            scc_seq_ok = False
    check('s_cselect_b32 after s_bitset1_b32/s_bitset0_b32 reads the earlier compare', scc_seq_ok)

    # 3. Constant folding and uniformity.
    fold_ok = True
    for op in OPS:
        for d_val in SPECIAL:
            for bit_idx in (0, 5, 31, 37, -1):
                _, st = translate(encode_sop1(op, REG, encode_inline(bit_idx)),
                                  setup=lambda s, v=d_val: s.sym.__setitem__(REG, ('const', v)))
                want = ref_bitset(op, d_val, bit_idx)
                if st.sym.get(REG) != ('const', want) or st.lines != ['s%d = 0x%08xu;' % (REG, want)]:
                    fold_ok = False
                    break
            if not fold_ok:
                break
        # Constant SGPR index also folds.
        _, st_cs = translate(encode_sop1(op, REG, SRC_REG), setup=lambda s: (
            s.sym.__setitem__(REG, ('const', 0x10)), s.sym.__setitem__(SRC_REG, ('const', 37))))
        want_cs = ref_bitset(op, 0x10, 37)
        if st_cs.sym.get(REG) != ('const', want_cs) or st_cs.lines != ['s%d = 0x%08xu;' % (REG, want_cs)]:
            fold_ok = False
    check('constant operands fold to exact 32-bit words without touching SCC', fold_ok)

    _, st_ud = translate(encode_sop1('s_bitset1_b32', REG, encode_inline(5)),
                         setup=lambda s: s.sym.__setitem__(REG, ('ud', 2)))
    check('ud destination drops ud symbol and stays uniform',
          st_ud.uniform_source('s%d' % REG) and st_ud.sym.get(REG) is None and st_ud.lines == [
              's5 = ud2 | (1u << (0x00000005u & 31u));'])
    _, st_u = translate(encode_sop1('s_bitset0_b32', REG, SRC_REG))
    check('uniform D and uniform S0 keep D and SCC uniform',
          st_u.uniform_source('s%d' % REG) and st_u.scc_uniform)
    _, st_vd = translate(encode_sop1('s_bitset1_b32', REG, SRC_REG),
                         setup=lambda s: s.varying_s.add(REG))
    check('varying D keeps D varying and leaves SCC uniform unchanged',
          not st_vd.uniform_source('s%d' % REG) and st_vd.scc_uniform)
    _, st_vs = translate(encode_sop1('s_bitset0_b32', REG, SRC_REG),
                         setup=lambda s: s.varying_s.add(SRC_REG))
    check('varying S0 marks D varying and leaves SCC uniform unchanged',
          not st_vs.uniform_source('s%d' % REG) and st_vs.scc_uniform)

    # 4. Refusals leave state intact and emit nothing.
    mask_refused = True
    for op in OPS:
        for setup, asm, want in (
            (lambda s: s.masks.add(REG), '%s s%d, 1' % (op, REG), '%s on a lane mask' % op),
            (lambda s: s.sym.__setitem__(REG, ('execcopy',)), '%s s%d, 1' % (op, REG), '%s on a lane mask' % op),
            (lambda s: s.masks.add(SRC_REG), '%s s%d, s%d' % (op, REG, SRC_REG), '%s on a lane mask' % op),
            (lambda s: s.sym.__setitem__(SRC_REG, ('execcopy',)), '%s s%d, s%d' % (op, REG, SRC_REG),
             '%s on a lane mask' % op),
            (None, '%s s%d, vcc_lo' % (op, REG), '%s on a lane mask' % op),
            (None, '%s s%d, exec_lo' % (op, REG), '%s on a lane mask' % op),
            (None, '%s vcc_lo, 1' % op, '%s destination vcc_lo' % op),
            (None, '%s exec_lo, 1' % op, '%s destination exec_lo' % op),
        ):
            msg, intact = refusal(op, asm, setup)
            if msg != want or not intact:
                mask_refused = False
    check('lane-mask destinations and sources refused without changing state', mask_refused)

    sym_refused = True
    for op in OPS:
        for kind in ('desc', 'descnum', 'ptr', 'ptrhi', 'smpheap', 'unassigned_const'):
            key = (('desc', ('table', 0), 0, 0) if kind == 'desc'
                   else ('const', 0, 'unassigned') if kind == 'unassigned_const' else (kind, 0))
            expected_kind = 'const' if kind == 'unassigned_const' else kind
            for target_reg, asm in ((REG, '%s s%d, 1' % (op, REG)), (SRC_REG, '%s s%d, s%d' % (op, REG, SRC_REG))):
                msg, intact = refusal(op, asm, lambda s, r=target_reg, k=key: s.sym.__setitem__(r, k))
                if msg != '%s on %s data' % (op, expected_kind) or not intact:
                    sym_refused = False
        msg_m0, intact_m0 = refusal(op, '%s s%d, m0' % (op, REG), lambda s: s.sym.__setitem__('m0', ('ptr', ('root', 0))))
        if msg_m0 != '%s on ptr data' % op or not intact_m0:
            sym_refused = False
    check('descriptor, pointer and unassigned-constant data refused on D, S0 and m0', sym_refused)

    malformed_refused = True
    for op in OPS:
        for asm in ('%s m0, 1' % op, '%s s[2:3], 1' % op, '%s v1, 1' % op, '%s s106, 1' % op,
                    '%s s5, s[2:3]' % op, '%s s5, v1' % op, '%s s5, s106' % op,
                    '%s s5, 1.0' % op, '%s s5, -0.5' % op, '%s s5, src_scc' % op, '%s s5, 0x100000000' % op,
                    '%s' % op, '%s s5' % op, '%s s5, 1, 2' % op, '%s s5, 1 clamp' % op, '%s s5, 1 mul:2' % op):
            msg, intact = refusal(op, asm)
            if msg is None or not intact:
                malformed_refused = False
    for wide in ('s_bitset0_b64', 's_bitset1_b64'):
        msg, intact = refusal(wide, '%s s[4:5], 1' % wide)
        if msg is None or not intact:
            malformed_refused = False
    check('malformed operands, modifiers and 64-bit forms stay refused', malformed_refused)

    # 5. Draw fixture word decoding & normal IEEE-754 output sanity.
    draw_ok = True
    for name, src, words, expected, mutated in DRAW_FIXTURES:
        decoded_asm = '\n'.join(asm for _, _, _, asm in G.decode(words)) + '\n'
        if decoded_asm != src:
            draw_ok = False
        for w in expected + (mutated or ()):
            if not is_normal_f32(w):
                draw_ok = False
    check('draw fixture bytes decode cleanly and output normal f32 words', draw_ok)
    return results


def load_mutant(old, new):
    source = Path(G.__file__).read_text(encoding='utf-8').replace('\r\n', '\n')
    assert source.count(old) == 1, old
    module = types.ModuleType('gfx2hlsl_mutant')
    module.__file__ = G.__file__
    exec(compile(source.replace(old, new), G.__file__, 'exec'), module.__dict__)
    return module


MUTATIONS = (
    ('swap set/clear in runtime expression',
     "expr = '%s | (1u << (%s & 31u))' % (old, bit) if set_bit else '%s & ~(1u << (%s & 31u))' % (old, bit)",
     "expr = '%s & ~(1u << (%s & 31u))' % (old, bit) if set_bit else '%s | (1u << (%s & 31u))' % (old, bit)"),
    ('drop & 31u bit-index mask',
     "expr = '%s | (1u << (%s & 31u))' % (old, bit) if set_bit else '%s & ~(1u << (%s & 31u))' % (old, bit)",
     "expr = '%s | (1u << %s)' % (old, bit) if set_bit else '%s & ~(1u << %s)' % (old, bit)"),
    ('swap set/clear in constant fold',
     'value = (d_const | mask) if set_bit else (d_const & ~mask)',
     'value = (d_const & ~mask) if set_bit else (d_const | mask)'),
    ('write SCC in scalar_bitset',
     'self.set_s(d, expr)\n        return None',
     "self.set_s(d, expr, scc='$ != 0u')\n        return None"),
    ('drop lane-mask refusal on D',
     "if d in getattr(self, 'masks', ()) or self.sym.get(d, ('',))[0] == 'execcopy':",
     'if False:'),
    ('drop lane-mask refusal on S0',
     'if self.is_lane_mask(src_tok):',
     'if False:'),
    ('drop descriptor/symbol refusal on D',
     "if sym and (sym[0] not in ('const', 'ud', 'fixed_buffer_word') or (sym[0] == 'const' and len(sym) > 2)):",
     'if False:'),
    ('remove op_s_bitset1_b32 lowering',
     'def op_s_bitset1_b32(self, ops, mods):',
     'def op_unused_bitset1_b32(self, ops, mods):'),
)


def controls():
    failures = []
    for label, ok, detail in battery(G):
        print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
        if not ok:
            failures.append(label)
            if detail:
                print('  ' + detail)
    for label, old, new in MUTATIONS:
        mutant = load_mutant(old, new)
        try:
            red = [result[0] for result in battery(mutant) if not result[1]]
        except Exception as error:
            red = ['%s: %s' % (type(error).__name__, error)]
        print('mutation %s: %s' % (label, 'caught by %d checks' % len(red) if red else 'NOT CAUGHT'), flush=True)
        if not red:
            failures.append('mutation not caught: ' + label)
    if failures:
        raise AssertionError('bitset controls failed:\n  ' + '\n  '.join(failures))


def check_binary(llvm_mc):
    sop1_src = '\n'.join(asm for asm, _ in ENCODINGS) + '\n'
    sop1_want = b''.join(bytes.fromhex(h) for _, h in ENCODINGS)
    if assemble(llvm_mc, sop1_src) != sop1_want:
        raise AssertionError('SOP1 encodings differ from llvm-mc')
    if assemble(llvm_mc, VS_SOURCE) != VS_WORDS:
        raise AssertionError('vertex fixture differs from llvm-mc')
    for name, src, words, _, _ in DRAW_FIXTURES:
        built = assemble(llvm_mc, src)
        if built != words:
            raise AssertionError('%s: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for name, _, words, expected, mutated in DRAW_FIXTURES:
                pipe = G.Pipeline(VS_WORDS, words, root_blob())
                vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_%s_%s' % (name, kind), vs_profile, directory)
                ps_hlsl = pipe.hlsl_ps()
                ps = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%s' % (name, kind), ps_profile, directory)
                want = words_text(expected)
                run = draw(renderer, adapter, vs, ps, want)
                if run.returncode == 77:
                    print(run.stdout, run.stderr)
                    sys.exit(77)
                if run.returncode:
                    raise AssertionError('%s %s\n%s\n%s' % (kind, name, run.stdout, run.stderr))
                print('%s %s: %s' % (kind, name, want), flush=True)
                if mutated is None:
                    continue
                wrong = words_text(mutated)
                bad_ps = compile_hlsl(compiler, kind, mutate_draw_hlsl(ps_hlsl),
                                      'mut_%s_%s' % (name, kind), ps_profile, directory)
                run = draw(renderer, adapter, vs, bad_ps, want)
                if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                    raise AssertionError('%s %s mutation did not read %s\n%s\n%s'
                                         % (kind, name, wrong, run.stdout, run.stderr))
                print('%s %s mutation: fails as expected, reads %s' % (kind, name, wrong), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
