# Fresh0 two-encoder MMD successor

Latest human instructions: finish full350 Qwen-Image-Bench, start the successor
from the official pretrained generator at0, replace SigLIP with Qwen-VL and remove
AIM from MMD. Training during judge download was permitted, then evaluation when
weights finish. Judge weights are now verified, so active350 evaluation keeps
priority; reference preparation runs concurrently in bounded spare memory. New
full-memory training follows actual evaluation completion. The old418 stop does
not pause this newly authorized successor and the old500 objective must not cause
the old experiment to resume. Both350 evaluation and new500 training remain work.

The successor config is `z-image/configs/zimage_2step_1024_dino_qwenvl_mmd2_from0.yaml`.
MMD encoders are exactly DINOv3 and Qwen3-VL; GAN retains its two1024px spatial
PatchGAN heads DINOv3/Qwen-VL. Text remains Qwen3-4B2560D. The existing MMD Qwen
adapter uses the pretrained vision tower+merger final-token mean2560D, with256px
internal input after the shared ADM512 preprocessing. The existing GAN Qwen
spatial feature contract, including its language-decoder path, is unchanged.
No new image-feature L2 normalization is introduced: raw image features and the
existing per-encoder RBF calibration are retained. Text remains L2 normalized.

`bash ops/h20-run-dino-qwenvl-reference.sh` builds a separate reference view from
the already retained125808 canonical1024px teacher PNGs. Every input PNG is
checked against the canonical index SHA; source stores remain immutable. Each
GPU4/5 worker has a3.5GiB allocator budget, FP32 frozen vision weights and BF16
compute, and commits disjoint float16 rows with resumable chunk hashes. The
encoder's finite nonzero pixel gradient and frozen parameters are checked once.
DINO same-row alignment keeps the existing relative<=.03/cosine>=.999 gates.
Publication requires full125808-row finite coverage, consistent contracts, a
fixed sigma=.25*median over2000 deterministic stored image rows, beta=sigma/s_txt,
finite required500-step/statistic text rows with unit norm, and valid reference
statistics. The view contains only DINO/Qwen image streams and shares the original
text, contexts, row order and prompt identities through verified links.

Generator base2.83e-5, D base1e-5, softplus, adaptive D policy, clip4, global32,
micro2/accum8, FA3 and checkpoint groups2 remain the current settings. Student
weights must retain every50..500 milestone with final-named SHA readback. The
old resume400 stays local. During judging, `h20-stage-judge-cache-offload.py` copies
the49.8GB first judge shard to shared object storage and verifies its final SHA.
The active source path remains untouched until actual full350 benchmark exit0,
all8 parts and4105 answers, and judge subprocess exit. Only then replace that
verified local cache payload with its remote symlink, recovering enough space
for the new rolling resume and a student snapshot plus margin. Never release an
unverified checkpoint. Require actual350 benchmark exit0, complete
new reference, sufficient remote checkpoint/cache space and a real two-GPU
D/G-active smoke gate. Reinitialize from official pretrained0 after smoke; do not
carry its updates or warm from350/400. Normal commits/main push and full Mac
code/docs backup are required; all weights/features/environments stay remote.
