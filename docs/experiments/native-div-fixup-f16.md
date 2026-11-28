# Direct DXIL `v_div_fixup_f16`

Lowers `v_div_fixup_f16` (VOP3 `0x35f`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) using exact 32-bit integer IEEE-754 binary16
division-fixup predicates while preserving `D[31:16]`.

## Scope

- Covers finite quotient sign propagation (`sign(S1) ^ sign(S2)`), `x/0` and
  `Inf/x` (`±Inf`), `0/x` and `x/Inf` (`±0.0h`), `0/0` and `Inf/Inf` (`0x7e00`),
  NaN-quotient overflow (`±Inf`), `S1`/`S2` NaN quieting (`| 0x0200`), inline
  binary16 float constant `0.5` (`src0=240`), VOP3 `abs`/`neg`/`clamp`, and
  `D[31:16]` preservation.
- Refuses unsupported modifiers (`omod`, `op_sel`).
- Negative controls: omitting `sign_out` XOR, omitting `0/0`/`Inf/Inf` invalid
  check, omitting `Q=NaN` overflow check, omitting `clamp`, reading inline `0.5`
  as `0x0000`, and zeroing `D[31:16]` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_div_fixup_f16_' -V
```
