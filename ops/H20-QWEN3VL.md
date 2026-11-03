# Qwen3-VL-4B vision shard prefetch

Run on the remote Linux node, using its existing HF/httpx environment:

```sh
python ops/h20-qwen3vl-prefetch.py --root /data/shaoshitong/rdm-sets
```

The script pins `Qwen/Qwen3-VL-4B-Instruct` to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. Official HF metadata supplies
the inventory and each file's size and SHA-256 or Git blob SHA-1. It requests
all JSON files and only `model-00002-of-00002.safetensors`; that shard must
be 3908490048 bytes with SHA-256
`046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6`.
The first weights shard is never requested, and no tensors/model are loaded.

Payload transport defaults to `https://hf-mirror.com`, with credentials disabled.
Official metadata uses the existing proxy configuration; payload transport
defaults to direct access. Use `--trust-env-transport` when the payload endpoint
needs the existing proxy, or `--endpoint https://huggingface.co` for official
transport. These options change neither the revision nor expected hashes.

HF's standard cache is `ROOT/hf/hub`. Interrupted transfers resume through its
normal incomplete files and locks when the same command runs again. JSON stdout
reports observed cache bytes every five seconds, followed by explicit per-file
hash results. Observed bytes are not a verification claim. The script refuses
payload execution on macOS and requires an existing absolute remote root.

`ROOT/control/qwen3vl-4b-prefetch-inventory.json` records the official metadata.
Only after every file passes size/hash checks and all 315 indexed visual keys
are confirmed in the shard header does the script atomically publish
`ROOT/control/qwen3vl-4b-vision-prefetch-verified.json`. The receipt records
the snapshot, digests and visual keys. It does not certify feature extraction,
reference coverage, image preprocessing, GPU gradients or training capacity.
An existing receipt describes its original verification time; re-running always
revalidates cached bytes before publishing a fresh receipt.
