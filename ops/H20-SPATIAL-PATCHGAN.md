# H20 spatial PatchGAN

Implements the Qwen3-VL / DINOv3 spatial PatchGAN Page specification
`page_3bd51524f3008191b740abb3cea05cd3`, including the requested concatenation
of the existing 2560-dimensional Qwen3-4B caption feature into each region.

Both GAN image branches receive the same 1024 RGB quantization for real and fake
images. Qwen3-VL-4B-Instruct uses the fixed instruction before the image, native
merger, image mask, causal attention and DeepStack; it executes only the first
24 language decoder blocks and reads their output before the final norm as a
`B×2560×32×32` map. DINOv3-L supplies the final normalized spatial patches,
without CLS/register tokens, as `B×1024×64×64`.

Each branch has a trainable nonoverlapping convolution (Qwen kernel/stride4,
DINO kernel/stride8), producing an 8×8 region grid. Its shared regional critic
concatenates the broadcast caption feature, applies bidirectional attention with
2D row/column RoPE, and returns 64 raw logits per image. Hinge losses average
over batch and region together, without an additional division by64 or2.
The original encoder1/3 and generation-step weights still apply.

The old global DINO/SigLIP/AIM MMD definitions, global reference features and
ADM512 preprocessing remain separate. The original AIM global GAN is retained.
Learning rates, schedule, GAN coefficient, adaptive coefficient formula, step
weights, global batch32 and checkpoint intervals stay in the saved config.
Adaptive gradient norms use decoded pixels as the common space for the global
MMD and spatial GAN. Two-level pixel GradCache replays encoders and generator
separately; frozen encoder weights still transmit image gradients. D updates
detach image features; G uses the updated, frozen critic and fixed spectral-norm
buffers. Aggregator outputs are never cached.

## Data and source roles

All large files and caches stay under `/data/shaoshitong/rdm-sets` on H20.
The complete Qwen3-VL snapshot is pinned to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. Teacher images are pinned to dataset
revision `3195cc9e58d83dfe17a6b5d89f9886121872586f`: 87 retained tars,
173074565120 bytes, and125808 canonical rows. The final manifest lives at
`zimage/zimage-ref-v1/reference_images_patchgan/manifest.json` under that root.
The index verifies complete PNG decoding, prompt/image identities, render seed
and settings, official tar hashes, and byte offsets. Its source-hash contract
must remain unchanged while the index is being assembled.

The asset downloader has its own published branch `codex/h20-patchgan-assets`.
It relays only HF metadata requests through the existing loopback18791 proxy;
signed CDN payload URLs are passed to aria2 on stdin, with per-URI `dir`/`out`,
and the payload downloads directly on H20. Originals, partials, failed receipts
and controlled replacement history are retained. The training/validation source
is published separately on `codex/h20-spatial-patchgan`.

Real spatial features are computed online from the canonical teacher images,
with a bounded CPU LRU. Generated spatial maps are cached only for the current
sliding window in BF16; feature-gradient leaves are FP32. A full frozen spatial
reference cache would require about1.715 TB and is not deployed.

## Bounded acceptance

`ops/h20-run-patchgan-validate.sh` uses only physical GPU4/5 and never starts
formal500-step A/B runs, saves their weights, or starts evaluation/W&B.
It runs actual-environment CPU tests, two-process Gloo normalization checks,
actual-weight single-image forward/backward checks, and then releases GPU
contexts and the run lock while waiting for the complete teacher-image manifest.
It subsequently checks an actual indexed teacher PNG and runs three complete
dual-GPU student iterations plus probe/refresh. The candidate uses FA3,
checkpoint groups of2 and microbatch2; only a verified CUDA OOM permits a
microbatch1 fallback at the same global batch32.

For this bounded test only, `g_start_step=0` exercises the generator GAN path
without burning50 formal warmup steps. The saved experiment configs keep their
original warmup. Each accepted iteration must have one D update per generation
step, no D/G skips, all three heads' D/G losses, finite positive D/G gradient
norms and per-step pixel adversarial norms. Timings exclude the first update
when reporting steady throughput. Logs retain per-rank CUDA allocation/reservation
and the controller records both cards' NVML peaks. Probe/refresh is checked too.

Final completion requires the full image manifest, actual-PNG GPU receipt,
three valid full iterations and probe/refresh, a current-source
`control/patchgan-acceptance.json`, and `control/patchgan-validate.exit=0`.
CPU tests or a single-image GPU check alone do not establish completion.
