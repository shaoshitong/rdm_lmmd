# Fixed complex portrait comparison

User chose to keep the current training and compare pretrained0 against actual150,
then retain the same cases at subsequent existing50-step saves. The overwritten50/100
weights are not reconstructed. Training code, configuration, sampling and save policy
are not modified by this workflow.

Official dataset: https://huggingface.co/datasets/Qwen/Qwen-Image-Bench
Revision: `d2493deb153b020cf169c7e3f57d15e4dd697038`, Apache-2.0.
`qib-portrait-cases-v1.json` contains eight exact English prompts, original bilingual
metadata and prompt hashes. Cases are selected before inference:5,8,47,81,172,312,393,849.
These qualitative previews do not constitute a full benchmark or judge score.

`h20-run-qib-portrait-preview.sh 150 6 --include-initial` renders16 PNGs per
stage at native1024 pixels, seeds0/1, identical prompt-ID-based latent noise,
two-step sigmas1/.75, renoise enabled, native attention, BF16 transformer and
VAE decode outside autocast. Prompt lengths including the original chat wrapper
are checked before inference so none is silently truncated at512 tokens.

Only a spare physicalGPU6 or7 can be selected; preflight checks48GiB free twice,
the CUDA allocator is capped at32GiB, a per-GPU exclusive preview lock is acquired,
and the inference command has a30minute timeout. Current training stays on4/5.
The rolling checkpoint is loaded with its actual step/schedule/head contract
checked; no checkpoint is copied, linked, overwritten or archived.

`h20-run-follow-qib-portrait-saves.sh` waits at most24h for actual existing
200/250/300/350/400/450/500 saves, checking each newly published checkpoint inode
and payload step. It waits for spareGPU6 headroom, invokes the same bounded preview,
and records true exit and image receipts. An overwritten stage is explicitly missed,
never relabeled. Failure stops this helper without touching training.

Remote artifacts: `/data/shaoshitong/rdm-sets/visualizations/twohead-lr10-A/qib-portraits-v1`.
Logs and true exits: `launch-logs/twohead-qib-portraits-*.log`,
`control/twohead-qib-portraits-*.exit`. Each comparison page is2048px wide;
each source tile remains1024x1024 and is pixel-compared after PNG save/readback.

`h20-plot-twohead-loss.py` plots only existing logged scalars, omits inactive
GAN-G warmup entries, retains raw spikes, overlays a trailing10-iteration mean,
and marks the recorded D learning-rate action at its effective iteration125.
PNG and vector PDF exports need no training hooks or extra model statistics.

## Retained-weight follow-up after the authorized200 resume

Every50-step student from200 onwards is now independently archived. The current
portrait follower waits for each stage's successful final-object SHA receipt and
passes that exact archive to `--checkpoint`. It never substitutes the latest rolling
resume or drops a requested case merely because GPU6 became available later.
The saved student has no critic tensors; the loader verifies its step/schedule,
world size, two-step sampling, all521 FP32 model tensors and the exact four adaptive
D head keys. This is a CPU metadata check against an already fully hashed object.
Direct object-store mmap metadata loading did not finish within90seconds in the
actual probe. The archived input is therefore read sequentially into the H20
process's RAM, SHA-checked again, deserialized on CPU, and released after loading.
There is no Mac or H20 disk staging copy. At least96GiB host/cgroup memory headroom
is required, and spare-GPU headroom is rechecked after the read. Loading byte counts
are explicitly distinct from generated-image progress.
The standby follower uses no Torch/GPU kernels. It preserves the existing0/150/200
previews and can catch up250..500 in order while training continues on4/5.
