# Direct DXIL `v_div_fixup_f32`

Lowers `v_div_fixup_f32` (VOP3 `0x15f`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) using exact 32-bit integer IEEE-754 binary32
division-fixup predicates.

## Scope

- Covers finite quotient sign propagation (`sign(S1) ^ sign(S2)`), `x/0` and
  `Inf/x` (`±Inf`), `0/x` and `x/Inf` (`±0.0`), `0/0` and `Inf/Inf` (`qNaN`),
  NaN-quotient overflow (`±Inf`), `S1`/`S2` NaN quieting, and VOP3 `abs`/`neg`/`clamp`.
- Refuses unsupported modifiers (`omod`, `op_sel`).
- Negative controls: omitting `sign_out` XOR, omitting `0/0`/`Inf/Inf` invalid
  check, omitting `Q=NaN` overflow check, and omitting `clamp` each differ from
  the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_div_fixup_f32_' -V
```
