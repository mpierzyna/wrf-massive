# TODO

## Streamline how env vars reach submitted jobs

`submit` passes the job environment explicitly via `sbatch --export=ALL,K=V,...`, because some clusters
(e.g. Snellius) drop variables that are only set in the environment of the sbatch process. Currently only
`WRF_MASSIVE_ENV_SHA256` and `FORCE_SETUP` (from `--force-setup`) are forwarded like this.

Other variables that `run` reads inside the job are not forwarded, so on such clusters they are silently lost:

- `WRF_MASSIVE_MONITOR` (memory/disk monitoring interval). Workaround used on Snellius (2026-11-19): take the
  `--debug submit` line and add `WRF_MASSIVE_MONITOR=<s>` to `--export` by hand.

Idea: one place that defines which `WRF_MASSIVE_*` variables `run` reads, and `submit` forwards every one of
them that is set (or a CLI option per variable, like `--force-setup`, e.g. `submit --monitor 30`). Keep the
comma check of `--export`.
