#!/usr/bin/env bash
# Batch profile (max_seqs=3), 1M context limit, experts + DSpark/MTP on GPU4.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export DSV41_LOCAL_PROFILE=batch
# Keep four dense stages; GPU4 also holds a routed-expert shard.
export DSV41_EP_DEVICES="${DSV41_EP_DEVICES:-2,0,1,3,4}"
export DSV41_EP_SHARDS="${DSV41_EP_SHARDS:-76,76,84,84,64}"
export DSV41_MTP_DRAFTS="${DSV41_MTP_DRAFTS:-4}"
case "$DSV41_MTP_DRAFTS" in
    3|4|5) ;;
    *) echo "Batch MTP requires DSV41_MTP_DRAFTS=3, 4, or 5" >&2; exit 2 ;;
esac
exec ./run_mtp_local.sh "$@"
