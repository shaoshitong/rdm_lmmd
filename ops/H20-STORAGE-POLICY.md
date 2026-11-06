# H20 project storage and naming

The 2026-11-06 user request supersedes personal-name directory defaults. The
canonical node workspace is `/data/rdm-workspace`; persistent new assets belong
under `/mnt/airs-business/rdm-experiments`. `/data/ready_ckpt_0423` is not ours and
must not be changed. Existing historical object archives keep their original
names and hashes; no foreign data is moved.

`ops/h20-storage-relocate.py` only runs after the actual 500-step training exit
and full final integrity receipt. It copies all 87 canonical image tars and the
final resume to object storage, verifies full SHA256 after final naming, and
retains durable per-file receipts before removing node copies. Original indices,
logs and failed attempts remain. The new `reference_images_patchgan_object_v1`
view preserves every image identity and offset, changing only paths/stat records
and the resulting SQLite hash. It tests PNG SHA reads from every tar. Model
weights are never transferred through the Mac.

The `.cache/qwen3vl-teacher-staging` name is historical: these 173 GB of tars are
actual GAN reference data, not disposable download cache. Future configs must
use the migrated object view. Object reads use bounded pread; they may be slower
than node-local reads. Stage into an explicitly budgeted local cache if needed.
Do not mmap model/resume files directly from the object mount. Verified student
checkpoints remain in their existing archive; a resume must be staged and SHA
verified onto an ordinary remote filesystem before mmap loading.

Keep frequently read contexts, model snapshots and the Python environment on
node storage. Preserve all 50-step students. Global `/data` availability is
shared with other users; measure it before launching and before checkpoint saves.
A cumulative process write counter is not retained disk usage.
