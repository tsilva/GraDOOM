# Changelog

## Unreleased

- Add matched cold-start Nature and small-ResNet deathmatch recipes with
  dual-provider player-kill evaluation, explicit checkpoint lineage, a 30-kill
  quality target, and a predeclared three-kill transfer margin.
- Pin the canonical 500M-transition recipes to the immutable published GradLab
  recipe (`v3`, 128 x 32, batch 256, learning rate `6.25e-5`) after rejecting a
  later mutable-checkout recipe whose scaled run plateaued near six kills.
- Record that mature 2,048 x 16 workload measurements made two concurrent jobs
  11.5% slower on the RTX 4090, while literal published-shape measurements make
  concurrent Nature and ResNet execution 45.1% faster in projected wall clock.
- Consume the renamed `env_vizdoom_turbo` provider and its exact
  `PLAYER_KILLCOUNT` signal while retaining map-wide `KILLCOUNT` only as the
  historical compatibility metric.
- Correct scratch-policy transfer evidence to use each policy's native-fused
  training renderer; Nature differs by only 0.03 mean player kills across the
  two providers, while small ResNet differs by 1.50.
- Bound reference-renderer polygon temporaries so 256-lane CUDA graph capture
  fits on the certified RTX 4090 without changing rendered output.
- Preserve explicit imported-evidence lineage when converting the immutable
  published GradLab checkpoint to the standalone format.
- Record the immutable published cold-start Nature policy's exact 33.25
  env-ViZDoom player-kill mean and its 17.39 reverse-transfer result in the
  Torch-native provider, without treating reverse transfer as certification.
- Migrate `EnvDoomTurboTorchVecEnv` to the breaking Turbo Vector API v2 shared
  constructor and defaults while keeping reset and step transitions Torch-only
  on `env.device`.
- Add immutable exact capabilities, portable signal schemas, numeric reset
  infos, deterministic catalog index zero, standardized async stepping, and
  opt-in RGB rendering.
- Reject non-neutral `num_threads` because the device path has no host worker
  pool.
- Preserve the certified deathmatch training, playback, CUDA smoke, and CUDA
  benchmark profiles through explicit construction settings.
- Point reference-provider documentation, diagnostics, and local asset defaults
  at the standardized `env-ViZDoom-turbo` project and `env-vizdoom-turbo`
  distribution names.
