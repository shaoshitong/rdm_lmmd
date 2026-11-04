# Authorized200 handoff and independent milestone weights

The user requested a resume after200 and independent weights every50, explicitly
including200,250,300,350,400. This supersedes the earlier500-only student retention.
Training still targets500 total updates on physical4/5, micro2, accumulation8.
GAN/MMD, LR schedules and adaptive-D behavior are unchanged.

`h20-stop-at200.py` waits for the actual200 save-completion line, reads the payload
using the existing CPU metadata auditor, checks the exact owned launcher identity,
records the current user authorization, and sends SIGTERM to that launcher only.
The resulting old-run signal exit is an intentional handoff, not an unexplained failure.

`h20-resume200-a.sh` verifies that handoff, checkpoint inode metadata, idle GPUs,
and unchanged model/algorithm files against the previously validated source. It resumes
the same data/cache/controller/D-Adam at200 to500, with `save_freq=50` and
`resume_every=50`. The old FSDP snapshot lacks G Adam; the user was explicitly told
that this authorized resume resets G Adam. No loss/LR/precision changes are bundled.

The local `/data` filesystem has approximately55GB free at handoff. Independent
FP32 students are about24.62GB each, so seven200..500 students cannot all stay there.
The remote mounted object-store destination is:

`/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A`

The mount passed actual64MiB write/fsync/rename/SHA readback. Its synthetic72PB
capacity from `df` is not a verified bucket quota and is never reported as such.

`h20-export-student200.py` exports the actual200 FP32 model, without cache/critics,
from the audited resume. Later students use the original trainer's checkpoint function.
`rdm.sw_lmmd.checkpoint_archive` works concurrently on CPU, validates each saved
model schema and step, copies to a private partial object, hashes during copy,
fully re-reads the remote SHA, publishes by rename, persists two receipts, and only
then removes that particular completed local student staging file. Resume checkpoints
are never deleted by the archiver. Existing conflicting destinations are never replaced.

Before any new save, the optional disk-space gate waits for archived staging files
to be released. It raises before writing if the worker failed or space cannot be
recovered within20minutes. The archiver closes the training-lock descriptor so an
archive process cannot accidentally keep the training lock after a later stop.

The first resume startup was rejected before any training iteration because the
existing FSDP guard requires `load_from == resume_from`. The corrected v2 entry
passes both paths, keeps that failed attempt's logs, and reuses its already-running
verified-destination archiver instead of launching a duplicate copy job.

A redundant tiny mount probe in the v2 preflight hit temporary ENOSPC while the
200 object was being archived. No new training iteration executed. After200 had
completed a full24.62GB verified archive and a fresh tiny probe passed, v3 uses
that stronger actual-archive receipt instead of repeating the redundant write.
`h20-verify-archived-weights.py` independently reads every final-named200..500 object
and checks its full SHA as well, with byte-progress and true-exit receipts.
The first independent final-object pass matched its complete SHA but rejected a
ks3fs mtime-cache refresh. The verifier now requires stable exact size plus the
complete expected content SHA; synthetic object-store inode/mtime changes alone
do not invalidate matching bytes. Local resume files retain strict inode/mtime checks.

Logs: `launch-logs/twohead-lr10-resume200-500-v3.log`,
`launch-logs/twohead-retain50-archive.log`.
True exits: `control/twohead-lr10-resume200-500-v3.exit`,
`control/twohead-retain50-archive/checkpoint-archive-worker.exit`.
Per-stage byte/SHA receipts: `control/twohead-retain50-archive/checkpoint-archive-N.json`.
All weights remain remote; Mac receives only source/docs and small audit summaries.

Read-only observer/auditor and portrait followers bind the resumed segment using
`H20_CONTROL_PREFIX`, `H20_TRAIN_SOURCE`, `H20_TRAIN_SHA` and the separate
`H20_PORTRAIT_FOLLOW_PREFIX`. Old signal exits and the failed startup remain in
their original logs. Cumulative iteration/counter checks still cover the single
continuous1..500 scalar log. The final student auditor accepts the authorized
object-store path only with its exact per-step archive receipt and full SHA match.
Its final archived-student pass uses the real200-validated sequential RAM loader,
which hashes the entire actual stream before deserialization, then scans every
saved floating tensor/scalar for finite values. Local rolling-resume audits continue
to use local mmap. Full scans still require actual500 and a clean training exit0.
Standalone CPU validators explicitly set `TMPDIR` to the existing remote
`/data/shaoshitong/rdm-sets/.cache/tmp` before importing Torch, because the node's
root filesystem has no usable default temporary space. Training already uses that
explicit remote temp directory; its environment is not changed by this fix.
