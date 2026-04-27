# FINESSE

Fidelity-aware quantum circuit transpilation for SNAIL-based superconducting architectures. Extends MIRAGE/LightSABRE with fidelity-weighted routing heuristics and mirror gate acceptance.

## Setup

```bash
git clone git@github.com:DylanJVA/MirageFidelities.git
cd MirageFidelities
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mqt]"
mkdir -p Results logs
```

For the Toronto (IBM heavy-hex) benchmark, also install:
```bash
pip install qiskit-ibm-runtime
```

## Running benchmarks

Single seed (for testing):
```bash
python FrequencyAllocationRuns.py --paper --seed 0
```

Full parallel run on a server (arguments: mode, seeds, parallel jobs):
```bash
nohup ./run_parallel.sh paper 20 $(nproc) > logs/master.log 2>&1 &
```

Toronto benchmark:
```bash
nohup ./run_parallel.sh toronto 20 $(nproc) > logs/master.log 2>&1 &
```

`nohup` keeps the job alive if your SSH session drops. Progress is logged to `logs/master.log` and per-seed logs in `logs/`.

Results land in `Results/paper.csv` and `Results/toronto.csv`. Figures are generated in `PaperPlots.ipynb`.
