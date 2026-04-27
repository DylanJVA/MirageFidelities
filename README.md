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

## Benchmark modes

| Flag | Description |
|---|---|
| `--paper` | 18-circuit paper suite on SNAIL topologies |
| `--toronto` | Same circuits on IBM FakeTorontoV2 (requires `qiskit-ibm-runtime`) |
| `--stress` | Denser QASMBench circuits for stress testing |
| `--quick` | 3 small circuits, fast sanity check |

Run with multiple seeds for statistical robustness (`--seeds N`). Each transpiler config is run independently per seed and the best result is post-selected:

```bash
python FrequencyAllocationRuns.py --paper --seeds 5
```

To run on a single specific circuit:
```bash
python FrequencyAllocationRuns.py --paper --circuit dnn_n16 --seeds 5
```

Results are saved to `Results/paper.csv` by default (`--output` to override). Figures are generated in `PaperPlots.ipynb`.

## Adding a new circuit

To benchmark a custom circuit, add it to `build_paper_circuits()` in `FrequencyAllocationRuns.py`:

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit.from_qasm_file("my_circuit.qasm")
# or build it programmatically
```

Then add a `("name", qc)` tuple to the list. The circuit will be routed on all configured topologies across all seeds.

## Running in parallel on a server

For large runs, each seed can be dispatched independently and merged afterward. `run_parallel.sh` handles this automatically (arguments: mode, seeds, parallel jobs):

```bash
nohup ./run_parallel.sh paper 20 $(nproc) > logs/master.log 2>&1 &
```

`nohup` keeps the job alive if your SSH session drops. Per-seed logs go to `logs/paper_s<N>.log`. Results are merged automatically into `Results/paper.csv` when all seeds finish.
