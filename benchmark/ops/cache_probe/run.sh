#!/bin/bash
# L2 (TCC) cache probe: spatial vs co-located at 4096^3.
# For each mode, for each counter (TCC_HIT, TCC_MISS — they don't fit one HW pass),
# launch 4 manual ranks with the in-process rocprofiler-sdk tool counting rank 0's
# layout kernel. Then print hit_rate = HIT/(HIT+MISS) per mode.
#
# Usage: bash run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROCM=/opt/rocm-7.2.4
TORCHLIB=/home/samanthe/tritonblas-work/.venv/lib/python3.12/site-packages/torch/lib
PY=/home/samanthe/tritonblas-work/.venv/bin/python
WORKER=$HERE/run_cache_probe.py

export LD_LIBRARY_PATH=$ROCM/lib:$TORCHLIB:${LD_LIBRARY_PATH:-}
export HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 WORLD_SIZE=4

run_one() {  # $1=mode $2=counter $3=outcsv
  local mode=$1 ctr=$2 out=$3
  rm -f "$out"
  # ranks 1..3: plain (no tool) to halve overhead; only rank 0 is counted
  for r in 1 2 3; do
    RANK=$r LOCAL_RANK=$r CACHE_MODE=$mode $PY $WORKER >/dev/null 2>&1 &
  done
  # rank 0: with the in-process tool
  RANK=0 LOCAL_RANK=0 CACHE_MODE=$mode \
    ROCPROFILER_LIBRARY_CTOR=1 \
    LD_PRELOAD=$ROCM/lib/librocprofiler-sdk.so \
    ROCP_TOOL_LIBRARIES=$HERE/tcc_tool.so \
    TCC_OUT=$out TCC_KERNEL_SUB=all_gather_matmul TCC_COUNTERS=$ctr \
    $PY $WORKER >/dev/null 2>&1
  wait
}

val() {  # $1=csv $2=counter -> value or 0
  awk -F, -v k="$2" '$1==k{print $2}' "$1" 2>/dev/null || echo 0
}

echo "=== L2 (TCC) cache probe: 4096^3 ws4, mtile fk16 ==="
printf "%-10s %12s %12s %10s\n" mode TCC_HIT TCC_MISS hit_rate
for mode in spatial coloc; do
  run_one $mode TCC_HIT  /tmp/tcc_${mode}_hit.csv
  run_one $mode TCC_MISS /tmp/tcc_${mode}_miss.csv
  H=$(val /tmp/tcc_${mode}_hit.csv TCC_HIT);  H=${H:-0}
  Mv=$(val /tmp/tcc_${mode}_miss.csv TCC_MISS); Mv=${Mv:-0}
  RATE=$(python3 -c "h=$H+0.0; m=$Mv+0.0; print(f'{h/(h+m):.4f}' if h+m>0 else 'n/a')")
  printf "%-10s %12s %12s %10s\n" "$mode" "$H" "$Mv" "$RATE"
done
