# FINESSE

Fidelity-aware quantum circuit transpilation for SNAIL-based superconducting architectures. Extends MIRAGE/LightSABRE with fidelity-weighted routing heuristics and mirror gate acceptance.

## Setup

```bash
git clone git@github.com:DylanJVA/MirageFidelities.git
cd MirageFidelities
python -m venv .venv && source .venv/bin/activate
pip install -e .
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

## Using FINESSE on your own circuit

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from finesse import route, prepare_dag

# Load your circuit
qc = QuantumCircuit.from_qasm_file("my_circuit.qasm")

# Define your device: coupling map and per-link fidelities
edges = [(0,1), (1,2), (2,3), (3,0)]
cm = CouplingMap(edges)

n = cm.size()
F = np.ones((n, n))   # uniform fidelity — replace with real calibration data
F[0,1] = F[1,0] = 0.99
F[1,2] = F[2,1] = 0.97
# ... etc

# Route
dag = prepare_dag(qc, cm)
routed_dag, n_swaps, final_layout = route(
    dag, cm,
    fidelity_matrix=F,
    aggression=2,        # FINESSE (mirror gates + fidelity)
    mode='lightsabre',
    seed=0,
)
```

`route()` returns the routed DAG, the number of SWAPs inserted, and the final logical→physical layout. To get a `QuantumCircuit` back, use `qiskit.converters.dag_to_circuit(routed_dag)`.

## Running in parallel on a server

For large runs, each seed can be dispatched independently and merged afterward. `run_parallel.sh` handles this automatically (arguments: mode, seeds, parallel jobs):

```bash
nohup ./run_parallel.sh paper 20 $(nproc) > logs/master.log 2>&1 &
```

`nohup` keeps the job alive if your SSH session drops. Per-seed logs go to `logs/paper_s<N>.log`. Results are merged automatically into `Results/paper.csv` when all seeds finish.
