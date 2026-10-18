#!/usr/bin/env bash
# Download DeepSeek-V4.1-Flash-1017 EXL3 2.75 bpw + the two official 1017 Engram shards into ONE folder.
#
#   bash scripts/download_model.sh [target-dir] [worker-host]
#
# target-dir   default: ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark  (needs ~420 GB free)
# worker-host  optional: after the download, copy the folder to the second Spark (same path) with rsync,
#              ideally over the direct fabric link (e.g. its 10.x address) instead of downloading twice.
#
# Safe to re-run: hf download skips finished files and resumes partial ones.
set -euo pipefail
DIR=${1:-$HOME/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark}
WORKER=${2:-}
QUANT=HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark
BASE=deepseek-ai/DeepSeek-V4.1-Flash-1017      # the 1017 shards! the older V4.1 has same-named files with other content
E1=model-00047-of-00048.safetensors; S1=101535150936
E2=model-00048-of-00048.safetensors; S2=101537926640

command -v hf >/dev/null || { echo "hf CLI missing: pip install -U 'huggingface_hub[cli,hf_xet]'"; exit 1; }
mkdir -p "$DIR"
free_gb=$(( $(df -Pk "$DIR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
have_gb=$(( $(du -sk "$DIR" | cut -f1) / 1024 / 1024 ))
[ $((free_gb + have_gb)) -ge 420 ] || { echo "need ~420 GB in $DIR (free ${free_gb} GB + already there ${have_gb} GB)"; exit 1; }

echo "== 1/2 quantized model (~199 GB) -> $DIR"
hf download "$QUANT" --local-dir "$DIR"
echo "== 2/2 Engram shards from $BASE (~203 GB) -> $DIR"
hf download "$BASE" "$E1" "$E2" --local-dir "$DIR"

for f in "$E1:$S1" "$E2:$S2"; do
  n=${f%%:*}; want=${f##*:}; got=$(stat -c %s "$DIR/$n")
  [ "$got" = "$want" ] || { echo "size mismatch for $n: $got (want $want)"; exit 1; }
done
echo "== done: $(ls "$DIR" | wc -l) files, $(du -sh "$DIR" | cut -f1) in $DIR"

if [ -n "$WORKER" ]; then
  echo "== copying to $WORKER:$DIR"
  ssh "$WORKER" "mkdir -p '$DIR'"
  rsync -a --info=progress2 --exclude .cache "$DIR/" "$WORKER:$DIR/"
fi
