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

| Flag | Fabric | Description |
|---|---|---|
| `--paper` | SNAIL (sqrt_iswap) | 18-circuit paper suite on 4 modular topologies |
| `--ibm` | IBM (CX/CZ) | Same circuits on Prague (33q), Brooklyn (65q), Washington (127q) |
| `--stress` | SNAIL | Denser QASMBench circuits for stress testing |
| `--quick` | SNAIL | 3 small circuits, fast sanity check |

`--ibm` requires `qiskit-ibm-runtime` and uses real IBM calibration data (heterogeneous fidelities from FakeProviderForBackendV2). The SNAIL modes use the physics-derived fidelity model from the paper.

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

Pass any QASM file directly — it runs through all configs and topologies, same as the paper suite:

```bash
python FrequencyAllocationRuns.py --qasm my_circuit.qasm --seeds 20
```

Results go to `Results/<circuit_name>.csv`. You can also filter to a specific topology:

```bash
python FrequencyAllocationRuns.py --qasm my_circuit.qasm --topology square_ring --seeds 20
```

### Python API

To use FINESSE directly in a script or notebook. For a custom SNAIL-like fabric:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from finesse import route, prepare_dag

qc = QuantumCircuit.from_qasm_file("my_circuit.qasm")

edges = [(0,1), (1,2), (2,3), (3,0)]
cm = CouplingMap(edges)

n = cm.size()
F = np.zeros((n, n))
F[0,1] = F[1,0] = 0.996   # per-link fidelities from physics model or calibration
F[1,2] = F[2,1] = 0.991
F[2,3] = F[3,2] = 0.987
F[3,0] = F[0,3] = 0.994

dag = prepare_dag(qc, cm)
routed_dag, n_swaps, final_layout = route(
    dag, cm,
    fidelity_matrix=F,
    basis_gate='sqrt_iswap',  # or 'cx' for IBM
    aggression=2,
    mode='lightsabre',
    seed=0,
)
```

For an IBM backend, pull fidelities directly from calibration data:

```python
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

backend = {b.name: b for b in FakeProviderForBackendV2().backends()}['fake_washington']
cm = backend.coupling_map
n  = backend.num_qubits
F  = np.zeros((n, n))
gate = 'cx' if 'cx' in backend.target.operation_names else 'cz'
for qargs, props in backend.target[gate].items():
    if props is not None and props.error is not None:
        i, j = qargs
        F[i, j] = F[j, i] = 1.0 - props.error

dag = prepare_dag(qc, cm)
routed_dag, n_swaps, final_layout = route(dag, cm, fidelity_matrix=F, basis_gate='cx')
```

`route()` returns the routed DAG, SWAP count, and final logical→physical layout. Convert back with `qiskit.converters.dag_to_circuit(routed_dag)`.

## Running in parallel on a server

For large runs, each seed can be dispatched independently and merged afterward. `run_parallel.sh <mode> <seeds> <jobs>` wraps the `--<mode>` flag from above — so `paper` corresponds to `--paper`, `ibm` to `--ibm`, etc.:

```bash
nohup ./run_parallel.sh paper 20 $(nproc) > logs/master.log 2>&1 &
```

`nohup` keeps the job alive if your SSH session drops. Per-seed logs go to `logs/paper_s<N>.log`. Results are merged automatically into `Results/paper.csv` when all seeds finish.
