# Original Turbo two/nine-step Qwen-Image-Bench

Latest human request: evaluate original Z-Image-Turbo at2 and9 steps using other
GPUs on this same H20 node. On2026-11-05 physical6/7 were measured idle,0MiB
compute use. Existing MMD2 training remains unchanged on4/5.

Entry: `bash ops/h20-run-qib-baselines.sh`. Independent lock/control prefix
`qib-turbo2-turbo9-gpu67`; never acquires the active training lock. Checks6/7
again immediately before loading; refuses occupied cards. No allocations on0..5.
No new model downloads or environment changes.

Run names `turbo2-zh-1024-seed0-v1` and `turbo9-zh-1024-seed0-v1`. Each has1000
official Chinese prompts, one1024x1024 PNG per expected ID, seed0, original
pretrained snapshot f332072aa78be7aecdf3ee76d5c247082da564a6 and native shifted
Euler schedule (`renoise=False`). The trained350 run uses `renoise=True`; the
native Turbo2 baseline is therefore distinct from pretrained0 previews using
the student's re-noising sampler. Text contexts/native BF16 model/VAE and
per-ID noise use the existing evaluator implementation.

Both cards split generation evenly, adaptive batch starts32, allocator cap95%,
finite pixels and atomic PNG writes. Generate and finish judging Turbo2, then
Turbo9. Each evaluation reuses the same pinned benchmark, Q-Judger, thinking
enabled, BF16, original judge.py and aggregation as350. Eight125-image shards,
two judge copies on6/7, batch80 (previously completed on this GPU model), bounded
OOM backoff only; no arbitrary inference restarts on observation timeouts.

Acceptance requires all1000 correct-resolution ID images, actual successful
generator/split/eight judge+thinking/merge/report exits,4105 raw and4105 parsed
L1 answers with exact metadata dimension sets, finite official scores, and a
full output archive final-named SHA readback on same-cluster object storage.
Source outputs remain remote; Mac only gets code/docs/small summaries.
Comparison report includes both original baselines and the verified350 run.

All subprocess commands/PIDs, source commit, real exits, failures/tracebacks and
stdout/stderr are durable under each run/logs and ROOT/control. Wrapper final
exit: ROOT/control/qib-turbo2-turbo9-gpu67.exit. Completed files/failed logs are
preserved; wrapper refuses a duplicate launch. No GPU occupation placeholders.
