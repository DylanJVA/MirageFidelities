#!/bin/bash
# Usage:
#   ./run_parallel.sh paper   [seeds] [jobs]
#   ./run_parallel.sh ibm [seeds] [jobs]
#
# Examples:
#   ./run_parallel.sh paper           # 20 seeds, all cores
#   ./run_parallel.sh paper 20 8      # 20 seeds, 8 at a time
#   ./run_parallel.sh ibm 20 8

MODE=${1:-paper}
SEEDS=${2:-20}
NJOBS=${3:-$(nproc)}

mkdir -p Results logs

echo "=== $MODE: seeds 0-$((SEEDS-1)), $NJOBS parallel slots, $(nproc) cores available ==="

pids=()

for seed in $(seq 0 $((SEEDS - 1))); do
    # Throttle: wait for a slot to open if we're at capacity
    while [ ${#pids[@]} -ge $NJOBS ]; do
        # Poll for any finished job
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $NJOBS ] && sleep 2
    done

    echo "  launching seed $seed ($(date +%H:%M:%S))"
    python FrequencyAllocationRuns.py --$MODE --seed $seed \
        > logs/${MODE}_s${seed}.log 2>&1 &
    pids+=($!)
done

# Wait for all remaining
echo "All seeds launched, waiting for completion..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "=== All seeds done ($(date +%H:%M:%S)) ==="

if [ "$MODE" = "paper" ]; then
    echo "Merging Results/paper_s*.csv → Results/paper.csv"
    python FrequencyAllocationRuns.py --merge
elif [ "$MODE" = "ibm" ]; then
    echo "Merging Results/ibm_s*.csv → Results/ibm.csv"
    python - <<'EOF'
import glob, pandas as pd, sys
files = sorted(glob.glob("Results/ibm_s*.csv"))
if not files:
    print("No per-seed files found."); sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["device","circuit","config","seed","wraparound"]).reset_index(drop=True)
df.to_csv("Results/ibm.csv", index=False)
print(f"Merged {len(files)} files → Results/ibm.csv ({len(df)} rows)")
EOF
fi

echo "Done. Check logs/ for per-seed output."
