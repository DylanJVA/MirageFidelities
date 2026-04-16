import copy
import csv
import os
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
def build_topology(wraparound=False):
    n = 32
    F_ring      = np.zeros((n, n))
    F_diag      = np.zeros((n, n))
    F_full      = np.zeros((n, n))
    for i in range(n):
        if i < n/2 - 1:
            F_ring[i,i+1]      = F_ring[i+1,i]      = 0.996
            F_diag[i,i+1]      = F_diag[i+1,i]      = 0.993
            F_full[i,i+1]      = F_full[i+1,i]      = 0.993
            F_diag[i, int(n/2)+i+1] = F_diag[int(n/2)+i+1, i] = 0.994
            F_full[i, int(n/2)+i+1] = F_full[int(n/2)+i+1, i] = 0.975
            if i > 0:
                F_full[i, int(n/2)+i-1] = F_full[int(n/2)+i-1, i] = 0.994
            if i & 2 == 0:
                F_ring[i,int(n/2)+i] = F_ring[int(n/2)+i,i] = .995
                F_diag[i,int(n/2)+i] = F_diag[int(n/2)+i,i] = .985
                F_full[i,int(n/2)+i] = F_full[int(n/2)+i,i] = .977
            else:
                F_ring[i,int(n/2)+i] = F_ring[int(n/2)+i,i] = .995
                F_diag[i,int(n/2)+i] = F_diag[int(n/2)+i,i] = .987
                F_full[i,int(n/2)+i] = F_full[int(n/2)+i,i] = .991
        elif i >= n//2 and i < n - 1:
            F_ring[i,i+1] = F_ring[i+1,i] = 0.994
            F_diag[i,i+1] = F_diag[i+1,i] = 0.986
            F_full[i,i+1] = F_full[i+1,i] = 0.989

    if wraparound:
        # connect tops
        F_ring[0,n//2-1]      = F_ring[n//2-1,0]      = 0.996
        F_diag[0,n//2-1]      = F_diag[n//2-1,0]      = 0.993
        F_full[0,n//2-1]      = F_full[n//2-1,0]      = 0.993
        # connect bottoms
        F_ring[n//2,n-1]      = F_ring[n-1,n//2]      = 0.994
        F_diag[n//2,n-1]      = F_diag[n-1,n//2]      = 0.986
        F_full[n//2,n-1]      = F_full[n-1,n//2]      = 0.989
        # first diagonal closing edge
        F_diag[n//2,n-1]      = F_diag[n-1,n//2]      = 0.994
        F_full[n//2,n-1]      = F_full[n-1,n//2]      = 0.975
        # last diagonal closing edge
        F_full[0,n-1]         = F_full[n-1,0]         = 0.994
        
    F_pentagon = np.zeros((38,38))
    
    F_pentagon[0,1]=F_pentagon[1,0]=0.96
    F_pentagon[1,2]=F_pentagon[2,1]=.962
    F_pentagon[2,3]=F_pentagon[3,2]=.968
    
    F_pentagon[3,4]=F_pentagon[4,3]=0.96
    F_pentagon[4,5]=F_pentagon[5,4]=.962
    F_pentagon[5,6]=F_pentagon[6,5]=.968
    
    F_pentagon[6,7]=F_pentagon[7,6]=0.96
    F_pentagon[7,8]=F_pentagon[8,7]=.962
    F_pentagon[8,9]=F_pentagon[9,8]=.968
    
    F_pentagon[9,10]=F_pentagon[10,9]=0.96
    F_pentagon[10,11]=F_pentagon[11,10]=.962
    F_pentagon[11,12]=F_pentagon[12,11]=.968
    
    F_pentagon[12,13]=F_pentagon[13,12]=0.96
    F_pentagon[13,14]=F_pentagon[14,13]=.962
    F_pentagon[14,15]=F_pentagon[15,14]=.968
    
    
    F_pentagon[16,17]=F_pentagon[17,16]=0.968
    F_pentagon[17,18]=F_pentagon[18,17]=.962
    F_pentagon[18,19]=F_pentagon[19,18]=.96
    
    F_pentagon[19,20]=F_pentagon[20,19]=0.968
    F_pentagon[20,21]=F_pentagon[21,20]=.962
    F_pentagon[21,22]=F_pentagon[22,21]=.96
    
    F_pentagon[22,23]=F_pentagon[23,22]=0.968
    F_pentagon[23,24]=F_pentagon[24,23]=.962
    F_pentagon[24,25]=F_pentagon[25,24]=.96
    
    F_pentagon[25,26]=F_pentagon[26,25]=0.968
    F_pentagon[26,27]=F_pentagon[27,26]=.962
    F_pentagon[27,28]=F_pentagon[28,27]=.96
    
    F_pentagon[28,29]=F_pentagon[29,28]=0.968
    F_pentagon[29,30]=F_pentagon[30,29]=.962
    F_pentagon[30,31]=F_pentagon[31,30]=.96
    
    F_pentagon[16,1] = F_pentagon[1,16]=.98
    F_pentagon[1,17] = F_pentagon[17,1] =.973
    
    F_pentagon[2,18] = F_pentagon[18,2]=.973
    F_pentagon[3,18] = F_pentagon[18,3] =.98
    
    F_pentagon[4,19] = F_pentagon[19,4]=.98
    F_pentagon[4,20] = F_pentagon[20,4] =.973
    
    F_pentagon[21,5] = F_pentagon[5,21]=.98
    F_pentagon[21,6] = F_pentagon[6,21] =.973
    
    F_pentagon[7,22] = F_pentagon[22,7]=.973
    F_pentagon[7,23] = F_pentagon[23,7] =.98
    
    F_pentagon[24,8] = F_pentagon[8,24]=.973
    F_pentagon[24,9] = F_pentagon[9,24] =.98
    
    F_pentagon[10,25] = F_pentagon[25,10]=.98
    F_pentagon[10,26] = F_pentagon[26,10] =.973
    
    F_pentagon[27,11] = F_pentagon[11,27]=.973
    F_pentagon[27,12] = F_pentagon[12,27] =.98
    
    F_pentagon[13,28] = F_pentagon[28,13]=.98
    F_pentagon[13,29] = F_pentagon[29,13] =.973
    
    F_pentagon[30,14] = F_pentagon[14,30]=.973
    F_pentagon[30,15] = F_pentagon[15,30] =.98
    
    F_pentagon[14,32] = F_pentagon[32,14]=.968
    F_pentagon[32,33] = F_pentagon[33,32] =.96
    
    F_pentagon[33,34] = F_pentagon[34,33] = .962
    F_pentagon[31,36] = F_pentagon[36,31]=.968
    F_pentagon[36,37] = F_pentagon[37,36] = .962
    
    F_pentagon[34,37] = F_pentagon[37,34]=.973
    if wraparound:
        F_pentagon[34,0] = F_pentagon[0,34] = .968
        F_pentagon[37,16] = F_pentagon[16,37] = .96
        F_pentagon[37,0] = F_pentagon[0,37] = .973

    cm_ring = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_ring[i,j] > 0])
    cm_diag = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_diag[i,j] > 0])
    cm_full = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_full[i,j] > 0])
    n_pent = 38
    cm_pentagon = CouplingMap([[i,j] for i in range(38) for j in range(38) if F_pentagon[i,j] > 0])
    return [
        ("square_ring",      cm_ring, F_ring),
        ("square_ring_diag", cm_diag, F_diag),
        ("square_ring_full", cm_full, F_full),
        ("pentagon_ring",    cm_pentagon, F_pentagon)
    ]

# ── Configs ───────────────────────────────────────────────────────────────────
EDGE_COST_W = 0.5

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
def run_circuits(circuit_list, n_seeds, label, out_path=None):
    """
    circuit_list: list of (name, QuantumCircuit | qasm_filename_str)
    Writes rows to out_path (default: results_{label}.csv) as they complete.
    Returns a DataFrame.
    """
    if out_path is None:
        out_path = f"results_{label}.csv"
    fieldnames = ["suite", "device", "circuit", "config", "seed", "swaps", "depth", "lf_cost"]

    # Open in append mode so partial results survive interruption.
    # Write header only if file is new/empty.
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    out_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

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

            # Only use qubits that have at least one edge — isolated nodes
            # (e.g. qubit 35 in pentagon_ring) cause the router to deadlock.
            connected = sorted({q for edge in cm.get_edges() for q in edge})

            for cfg_name, kwargs in configs:
                kw = dict(kwargs)
                if cfg_name in FIDELITY_CONFIGS:
                    kw["fidelity_matrix"] = F
                for seed in range(n_seeds):
                    rng = np.random.default_rng(seed + 10_000)
                    perm = rng.permutation(len(connected)).tolist()
                    initial_cur = [connected[p] for p in perm]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        routed, _, _ = route(
                            copy.deepcopy(dag_phys), cm,
                            seed=seed, initial_cur=initial_cur, **kw,
                        )
                    row = dict(
                        suite=label, device=dev_name, circuit=circ_name,
                        config=cfg_name, seed=seed,
                        swaps=swap_count(routed),
                        depth=routed.depth(),
                        lf_cost=circuit_lf_cost(routed, F),
                    )
                    writer.writerow(row)
                    out_file.flush()
                    rows.append(row)
            print(f"  [{label}] {dev_name} / {circ_name} done")

    out_file.close()
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
parser.add_argument("--test",       action="store_true", help="Smoke test: 1 seed, 1 circuit per suite")
parser.add_argument("--suite",      choices=["redqueen", "ext_small", "ext_large", "all"], default="all",
                    help="Which suite to run (default: all)")
parser.add_argument("--output",     default=None,
                    help="Output filename (without .csv). Default: results_{suite}. "
                         "Only valid when running a single suite.")
parser.add_argument("--circuit",    default=None,
                    help="Run only this circuit (e.g. ising_n26). Must be used with --suite.")
parser.add_argument("--seeds",      type=int, default=None,
                    help="Override number of seeds (default: per-suite value)")
parser.add_argument("--wraparound", action="store_true",
                    help="Add wraparound endpoint edges to the topology")
parser.add_argument("--topology",   default="all",
                    help="Comma-separated topology names to run, or 'all' (default). "
                         "Options: square_ring, square_ring_diag, square_ring_full, pentagon_ring")
args = parser.parse_args()

if args.output and args.suite == "all":
    parser.error("--output can only be used with a specific --suite, not --suite all")
if args.circuit and args.suite == "all":
    parser.error("--circuit requires a specific --suite")

all_devices = build_topology(wraparound=args.wraparound)
if args.topology == "all":
    devices = all_devices
else:
    requested = {t.strip() for t in args.topology.split(",")}
    devices = [(name, cm, F) for name, cm, F in all_devices if name in requested]
    if not devices:
        parser.error(f"No matching topologies found. Available: {[d[0] for d in all_devices]}")
if args.wraparound:
    print("Topology: wraparound enabled")
print(f"Running on topologies: {[d[0] for d in devices]}")

suites = {
    "redqueen":  (redqueen_circuits, 20),
    "ext_small": (ext_small,         10),
    "ext_large": (ext_large,          5),
}
to_run = ["redqueen", "ext_small", "ext_large"] if args.suite == "all" else [args.suite]

for name in to_run:
    circuits_for_suite, n_seeds = suites[name]
    if args.seeds is not None:
        n_seeds = args.seeds
    if args.circuit:
        circuits_for_suite = [(n, s) for n, s in circuits_for_suite if n == args.circuit]
        if not circuits_for_suite:
            parser.error(f"Circuit '{args.circuit}' not found in suite '{name}'")
    if args.test:
        circuits_for_suite = circuits_for_suite[:1]
        n_seeds = 1
        print(f"=== TEST: {name} (1 seed, 1 circuit) ===")
    else:
        print(f"=== {name} (n_seeds={n_seeds}{', wraparound' if args.wraparound else ''}) ===")
    out_path = f"{args.output}.csv" if args.output else None
    df = run_circuits(circuits_for_suite, n_seeds=n_seeds, label=name, out_path=out_path)
    print(df.groupby(["device","config"])[["swaps","depth","lf_cost"]].mean().round(2))
