"""Direct compute VOP3 v_div_fixup_f32 (exact IEEE-754 binary32 division fixup)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

# (S0=quotient, S1=denominator, S2=numerator)
CASES = (
    (0x3F000000, 0xC0000000, 0x3F800000),  # Q=+0.5, D=-2.0, N=+1.0 -> -0.5
    (0xBF400000, 0xC0800000, 0xC0400000),  # Q=-0.75, D=-4.0, N=-3.0 -> +0.75
    (0x3F800000, 0x80000000, 0x3F800000),  # D=-0.0, N=+1.0 -> -Inf
    (0x3F800000, 0x7F800000, 0xBF800000),  # D=+Inf, N=-1.0 -> -0.0
    (0x3F800000, 0x80000000, 0x00000000),  # 0/0 -> qNaN (0x7FC00000)
    (0x3F800000, 0xFF800000, 0x7F800000),  # Inf/Inf -> qNaN (0x7FC00000)
    (0x7FC00001, 0xC0000000, 0x3F800000),  # Q=NaN (overflow), D=-2.0, N=+1.0 -> -Inf
    (0x3F800000, 0x7F800001, 0x3F800000),  # D=sNaN -> quiet(D) (0x7FC00001)
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop3_ternary(op, vdst, src0, src1, src2, *, abs_m=0, neg_m=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs_m & 7) << 8) | (vdst & 0xFF))
    w1 = (((neg_m & 7) << 29) | ((omod & 3) << 27) |
          ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF))
    return [w0, w1]


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (s0, s1, s2) in enumerate(CASES):
        code += v_mov_lit(1, s0) + v_mov_lit(2, s1) + v_mov_lit(3, s2)
        # 0: plain v_div_fixup_f32 v4, v1, v2, v3
        code += vop3_ternary(0x15F, 4, 257, 258, 259) + [0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: negated denominator: v_div_fixup_f32 v4, v1, -v2, v3
        code += vop3_ternary(0x15F, 4, 257, 258, 259, neg_m=2) + [0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: abs denominator: v_div_fixup_f32 v4, v1, |v2|, v3
        code += vop3_ternary(0x15F, 4, 257, 258, 259, abs_m=2) + [0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: clamped: v_div_fixup_f32 v4, v1, v2, v3 clamp
        code += vop3_ternary(0x15F, 4, 257, 258, 259, clamp=1) + [0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def div_fixup_f32(s0, s1, s2, *, clamp=False, ignore_sign_xor=False, ignore_invalid=False, ignore_q_nan=False):
    m0, m1, m2 = s0 & 0x7FFFFFFF, s1 & 0x7FFFFFFF, s2 & 0x7FFFFFFF
    sign_out = (s0 & 0x80000000) if ignore_sign_xor else ((s1 ^ s2) & 0x80000000)
    if m1 > 0x7F800000:
        res = s1 | 0x00400000
    elif m2 > 0x7F800000:
        res = s2 | 0x00400000
    elif (not ignore_invalid) and ((m1 == 0 and m2 == 0) or (m1 == 0x7F800000 and m2 == 0x7F800000)):
        res = 0x7FC00000
    elif m1 == 0 or m2 == 0x7F800000:
        res = sign_out | 0x7F800000
    elif m1 == 0x7F800000 or m2 == 0:
        res = sign_out
    elif (not ignore_q_nan) and m0 > 0x7F800000:
        res = sign_out | 0x7F800000
    else:
        res = sign_out | m0
    if clamp:
        if (res & 0x7FFFFFFF) > 0x7F800000 or (res & 0x80000000):
            res = 0
        elif res > 0x3F7FFFFF:
            res = 0x3F800000
    return res & MASK


def reference(*, ignore_sign_xor=False, ignore_invalid=False, ignore_q_nan=False, no_clamp=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for s0, s1, s2 in CASES:
            r0 = div_fixup_f32(s0, s1, s2, ignore_sign_xor=ignore_sign_xor,
                               ignore_invalid=ignore_invalid, ignore_q_nan=ignore_q_nan)
            r_neg = div_fixup_f32(s0, s1 ^ 0x80000000, s2, ignore_sign_xor=ignore_sign_xor,
                                  ignore_invalid=ignore_invalid, ignore_q_nan=ignore_q_nan)
            r_abs = div_fixup_f32(s0, s1 & 0x7FFFFFFF, s2, ignore_sign_xor=ignore_sign_xor,
                                  ignore_invalid=ignore_invalid, ignore_q_nan=ignore_q_nan)
            r_clamp = div_fixup_f32(s0, s1, s2, clamp=not no_clamp, ignore_sign_xor=ignore_sign_xor,
                                    ignore_invalid=ignore_invalid, ignore_q_nan=ignore_q_nan)
            words.extend((r0, r_neg, r_abs, r_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert '2139095040' in dump and '2143289344' in dump, dump
    for tag, bad in (
        ('omod', vop3_ternary(0x15F, 4, 257, 258, 259, omod=1) + [0xBF810000]),
        ('op_sel', vop3_ternary(0x15F, 4, 257, 258, 259, op_sel=1) + [0xBF810000]),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and 'unsupported v_div_fixup_f32 operand modifiers' in (
            bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(ignore_sign_xor=True)
    assert reference() != reference(ignore_invalid=True)
    assert reference() != reference(ignore_q_nan=True)
    assert reference() != reference(no_clamp=True)
    print('v_div_fixup_f32 DXIL verified; invalid modifiers refused; sign, invalid, q-NaN and clamp controls differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    expected = reference()
    actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    if mismatch is not None:
        raise AssertionError(f'word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
    print('v_div_fixup_f32: 2048 result words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_div_fixup_f32_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
