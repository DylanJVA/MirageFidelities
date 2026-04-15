import copy
import warnings

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap

from finesse import (
    apply_trivial_layout, fetch_qasm, fetch_qasmbench,
    circuit_lf_cost, swap_count,
)
from finesse.routing import route

# ── Topology ──────────────────────────────────────────────────────────────────
n = 32
F_square_ring      = np.zeros((n, n))
F_square_ring_diag = np.zeros((n, n))
F_square_ring_full = np.zeros((n, n))
for i in range(n):
    if i < n/2 - 1:
        # top ring edges
        F_square_ring[i,i+1]      = F_square_ring[i+1,i]      = 0.996
        F_square_ring_diag[i,i+1] = F_square_ring_diag[i+1,i] = 0.993
        F_square_ring_full[i,i+1] = F_square_ring_full[i+1,i] = 0.993
        # top-left → bottom-right diagonal
        F_square_ring_diag[i, int(n/2)+i+1] = F_square_ring_diag[int(n/2)+i+1, i] = 0.994
        F_square_ring_full[i, int(n/2)+i+1] = F_square_ring_full[int(n/2)+i+1, i] = 0.975
        # top-right → bottom-left diagonal (skip i=0, no qubit n/2-1)
        if i > 0:
            F_square_ring_full[i, int(n/2)+i-1] = F_square_ring_full[int(n/2)+i-1, i] = 0.994
        # alternate rung fidelities
        if i & 2 == 0:
            # left
            F_square_ring[i,int(n/2)+i]      = F_square_ring[int(n/2)+i,i]      = .995
            F_square_ring_diag[i,int(n/2)+i] = F_square_ring_diag[int(n/2)+i,i] = .985
            F_square_ring_full[i,int(n/2)+i] = F_square_ring_full[int(n/2)+i,i] = .977
        else:
            # right
            F_square_ring[i,int(n/2)+i]      = F_square_ring[int(n/2)+i,i]      = .995
            F_square_ring_diag[i,int(n/2)+i] = F_square_ring_diag[int(n/2)+i,i] = .987
            F_square_ring_full[i,int(n/2)+i] = F_square_ring_full[int(n/2)+i,i] = .991
    elif i >= n//2 and i < n - 1:
        # bottom ring edges
        F_square_ring[i,i+1]      = F_square_ring[i+1,i]      = 0.994
        F_square_ring_diag[i,i+1] = F_square_ring_diag[i+1,i] = 0.986
        F_square_ring_full[i,i+1] = F_square_ring_full[i+1,i] = 0.989
# connecting the endpoints
# connect tops
F_square_ring[0,n//2-1]      = F_square_ring[n//2-1,0]      = 0.996
F_square_ring_diag[0,n//2-1] = F_square_ring_diag[n//2-1,0] = 0.993
F_square_ring_full[0,n//2-1] = F_square_ring_full[n//2-1,0] = 0.993
# connect bottoms
F_square_ring[n//2,n-1]      = F_square_ring[n-1,n//2]      = 0.994
F_square_ring_diag[n//2,n-1] = F_square_ring_diag[n-1,n//2] = 0.986
F_square_ring_full[n//2,n-1] = F_square_ring_full[n-1,n//2] = 0.989
# first diagonal closing edge
F_square_ring_diag[n//2,n-1] = F_square_ring_diag[n-1,n//2] = 0.994
F_square_ring_full[n//2,n-1] = F_square_ring_full[n-1,n//2] = 0.975
# last diagonal closing edge
F_square_ring_full[0,n-1] = F_square_ring_full[n-1,0] = 0.994
cm_square_ring      = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_square_ring[i,j]      > 0])
cm_square_ring_diag = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_square_ring_diag[i,j] > 0])
cm_square_ring_full = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_square_ring_full[i,j] > 0])

# ── Configs ───────────────────────────────────────────────────────────────────
EDGE_COST_W = 0.5

devices = [
    ("square_ring",      cm_square_ring,      F_square_ring),
    ("square_ring_diag", cm_square_ring_diag, F_square_ring_diag),
    ("square_ring_full", cm_square_ring_full, F_square_ring_full),
]

configs = [
    ("Standard SABRE",  dict(mode="sabre", aggression=0)),
    ("Standard MIRAGE", dict(mode="sabre", aggression=2)),
    ("FASST",           dict(mode="sabre", aggression=0,  edge_cost_weight=EDGE_COST_W)),
    ("FINESSE",         dict(mode="sabre", aggression=2,  fidelity_mirror=True, edge_cost_weight=EDGE_COST_W)),
]
FIDELITY_CONFIGS = {"FASST", "FINESSE"}

# ── Circuit helpers ───────────────────────────────────────────────────────────
def make_qaoa(n_qubits, p_layers):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    for _ in range(p_layers):
        for i, j in edges:
            qc.cx(i, j); qc.rz(0.5, j); qc.cx(i, j)
        qc.rx(0.3, range(n_qubits))
    return qc

# ── Benchmark runner ──────────────────────────────────────────────────────────
def run_circuits(circuit_list, n_seeds, label):
    """
    circuit_list: list of (name, QuantumCircuit | qasm_filename_str)
    Returns a DataFrame.
    """
    rows = []
    for dev_name, cm, F in devices:
        n_phys = cm.size()
        for circ_name, circ_src in circuit_list:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if isinstance(circ_src, str):
                    qc = fetch_qasm(circ_src)
                else:
                    qc = circ_src
                if qc.num_qubits > n_phys:
                    print(f"  skip {circ_name} ({qc.num_qubits}q > {n_phys}q)")
                    continue
                dag_phys = apply_trivial_layout(qc, cm)

            for cfg_name, kwargs in configs:
                kw = dict(kwargs)
                if cfg_name in FIDELITY_CONFIGS:
                    kw["fidelity_matrix"] = F
                for seed in range(n_seeds):
                    initial_cur = np.random.default_rng(seed + 10_000).permutation(n_phys).tolist()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        routed, _, _ = route(
                            copy.deepcopy(dag_phys), cm,
                            seed=seed, initial_cur=initial_cur, **kw,
                        )
                    rows.append(dict(
                        suite=label, device=dev_name, circuit=circ_name,
                        config=cfg_name, seed=seed,
                        swaps=swap_count(routed),
                        depth=routed.depth(),
                        lf_cost=circuit_lf_cost(routed, F),
                    ))
            print(f"  [{label}] {dev_name} / {circ_name} done")

    return pd.DataFrame(rows)


# ── Circuit suites ────────────────────────────────────────────────────────────
redqueen_circuits = [
    ("cm152a_212", "cm152a_212.qasm"),
    ("wim_266",    "wim_266.qasm"),
    ("dc1_220",    "dc1_220.qasm"),
    ("squar5_261", "squar5_261.qasm"),
    ("adr4_197",   "adr4_197.qasm"),
    ("cm42a_207",  "cm42a_207.qasm"),
]

# Split extended suite by size so large circuits use fewer seeds
ext_small = [
    ("adder_n10",      fetch_qasmbench("adder_n10",      size="small")),
    ("multiplier_n15", fetch_qasmbench("multiplier_n15", size="medium")),
    ("qft_n18",        fetch_qasmbench("qft_n18",        size="medium")),
    ("bv_n19",         fetch_qasmbench("bv_n19",         size="medium")),
    ("random_n20_d100",random_circuit(20, 100, max_operands=2, seed=42)),
    ("qaoa_n20_p2",    make_qaoa(20, 2)),
]

ext_large = [
    ("ising_n26",      fetch_qasmbench("ising_n26", size="medium")),
    ("vqe_n24",        fetch_qasmbench("vqe_n24",   size="medium")),
    ("qaoa_n25_p3",    make_qaoa(25, 3)),
    ("random_n32_d200",random_circuit(32, 200, max_operands=2, seed=42)),
]

# ── Run ───────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test",  action="store_true", help="Smoke test: 1 seed, 1 circuit per suite")
parser.add_argument("--suite", choices=["redqueen", "ext_small", "ext_large", "all"], default="all",
                    help="Which suite to run (default: all)")
args = parser.parse_args()

suites = {
    "redqueen":  (redqueen_circuits, 20),
    "ext_small": (ext_small,         10),
    "ext_large": (ext_large,          5),
}
to_run = ["redqueen", "ext_small", "ext_large"] if args.suite == "all" else [args.suite]

for name in to_run:
    circuits_for_suite, n_seeds = suites[name]
    if args.test:
        circuits_for_suite = circuits_for_suite[:1]
        n_seeds = 1
        print(f"=== TEST: {name} (1 seed, 1 circuit) ===")
    else:
        print(f"=== {name} (n_seeds={n_seeds}) ===")
    df = run_circuits(circuits_for_suite, n_seeds=n_seeds, label=name)
    df.to_csv(f"results_{name}.csv", index=False)
    print(df.groupby(["device","config"])[["swaps","lf_cost"]].mean().round(2))
