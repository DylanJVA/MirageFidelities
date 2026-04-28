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


def _mqt(name, n, level="indep"):
    """Fetch a circuit from MQT Bench. level: 'alg' or 'indep'."""
    from mqt.bench import get_benchmark, BenchmarkLevel
    lvl = BenchmarkLevel.INDEP if level == "indep" else BenchmarkLevel.ALG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_benchmark(name, lvl, n)


def build_paper_circuits():
    """Balanced 18-circuit suite: 6 per size band (8–14q, 15–21q, 22–32q)."""
    return [
        # 8–14 qubits (6)
        ("qpeexact_n8",       _mqt("qpeexact",      8, "alg")),
        ("wstate_n8",         _mqt("wstate",         8, "alg")),
        ("qft_n10",           _mqt("qft",           10, "indep")),
        ("ae_n10",            _mqt("ae",            10, "alg")),
        ("ghz_n10",           _mqt("ghz",           10, "alg")),
        ("vqe_two_local_n10", _mqt("vqe_two_local", 10, "alg")),
        # 15–21 qubits (6)
        ("seca_n11",          fetch_qasmbench("seca_n11",        size="medium")),
        ("multiplier_n15",    fetch_qasmbench("multiplier_n15",  size="medium")),
        ("dnn_n16",           fetch_qasmbench("dnn_n16",         size="medium")),
        ("qec9xz_n17",        fetch_qasmbench("qec9xz_n17",      size="medium")),
        ("square_root_n18",   fetch_qasmbench("square_root_n18", size="medium")),
        ("bv_n19",            fetch_qasmbench("bv_n19",          size="medium")),
        # 22–32 qubits (6)
        ("qft_n24",           _mqt("qft",           24, "indep")),
        ("qaoa_n25_p3",       make_qaoa(25, 3)),
        ("ising_n26",         fetch_qasmbench("ising_n26",       size="medium")),
        ("qft_n32",           _mqt("qft",           32, "indep")),
        ("qaoa_n32_p3",       make_qaoa(32, 3)),
        ("random_n32_d50",    random_circuit(32, 50, max_operands=2, seed=42)),
    ]


def build_stress_circuits():
    """11 medium QASMBench circuits (11–23 qubits) for stress testing.
    Skipped: sat_n11/bigadder_n18 (0 2Q gates), multiply_n13 (4 2Q),
             qft_n18 (306 2Q, slow), cc_n12 (classical if_else, not routable),
             bwt_n21 (21k 2Q, depth 53k), vqe_n24 (1.5M 2Q).
    """
    return [
        ("seca_n11",       fetch_qasmbench("seca_n11",       size="medium")),  # 11q, 36 2Q
        ("bv_n14",         fetch_qasmbench("bv_n14",         size="medium")),  # 14q, 13 2Q
        # adder_n10 dropped (trivial, already in paper suite)
        ("qf21_n15",       fetch_qasmbench("qf21_n15",       size="medium")),  # 15q, 46 2Q
        ("multiplier_n15", fetch_qasmbench("multiplier_n15", size="medium")),  # 15q, 30 2Q
        ("dnn_n16",        fetch_qasmbench("dnn_n16",        size="medium")),  # 16q, 384 2Q
        ("qec9xz_n17",     fetch_qasmbench("qec9xz_n17",     size="medium")),  # 17q, 32 2Q
        ("square_root_n18",fetch_qasmbench("square_root_n18",size="medium")),  # 18q, 118 2Q
        ("bv_n19",         fetch_qasmbench("bv_n19",         size="medium")),  # 19q, 18 2Q
        ("qram_n20",       fetch_qasmbench("qram_n20",       size="medium")),  # 20q, 16 2Q
        ("cat_state_n22",  fetch_qasmbench("cat_state_n22",  size="medium")),  # 22q, 21 2Q
        ("ghz_state_n23",  fetch_qasmbench("ghz_state_n23",  size="medium")),  # 23q, 22 2Q
    ]

# ── IBM fake topologies ───────────────────────────────────────────────────────
def _ibm_topology(backend_name: str):
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backends = {b.name: b for b in FakeProviderForBackendV2().backends()}
        backend = backends[backend_name]
    cm = backend.coupling_map
    n  = backend.num_qubits
    F  = np.zeros((n, n))
    gate = 'cx' if 'cx' in backend.target.operation_names else 'cz'
    for qargs, props in backend.target[gate].items():
        if props is not None and props.error is not None:
            i, j = qargs
            F[i, j] = F[j, i] = 1.0 - props.error
    return (backend_name, cm, F)

def build_ibm_topologies():
    """Three IBM backends spanning low→high fidelity heterogeneity.

    fake_toronto:    27q, std=0.024, range=0.122 — small, heterogeneous
    fake_sydney:     27q, std=0.027, range=0.146 — small, most heterogeneous
    fake_washington: 127q, std=0.017, range=0.098 — large, heterogeneous
    Note: circuits with more qubits than the backend are skipped automatically.
    """
    return [_ibm_topology(name) for name in
            ("fake_toronto", "fake_sydney", "fake_washington")]


# ── Topology ──────────────────────────────────────────────────────────────────
def build_topology(wraparound=False):
    # ── Square topologies (32 qubits: top row 0-15, bottom row 16-31) ─────────
    n = 32
    F_ring      = np.zeros((n, n))
    F_diag      = np.zeros((n, n))
    F_full      = np.zeros((n, n))
    for i in range(n):
        if i < n//2 - 1:       # top row edges + rungs (i=0..14)
            F_ring[i,i+1]      = F_ring[i+1,i]      = 0.996
            F_diag[i,i+1]      = F_diag[i+1,i]      = 0.993
            F_full[i,i+1]      = F_full[i+1,i]      = 0.993
            F_diag[i, n//2+i+1] = F_diag[n//2+i+1, i] = 0.994
            F_full[i, n//2+i+1] = F_full[n//2+i+1, i] = 0.975
            if i > 0:
                F_full[i, n//2+i-1] = F_full[n//2+i-1, i] = 0.994
            if i % 2 == 0:
                F_ring[i, n//2+i] = F_ring[n//2+i, i] = .995
                F_diag[i, n//2+i] = F_diag[n//2+i, i] = .985
                F_full[i, n//2+i] = F_full[n//2+i, i] = .977
            else:
                F_ring[i, n//2+i] = F_ring[n//2+i, i] = .995
                F_diag[i, n//2+i] = F_diag[n//2+i, i] = .987
                F_full[i, n//2+i] = F_full[n//2+i, i] = .991
        elif i >= n//2 and i < n - 1:   # bottom row edges (i=16..30)
            F_ring[i,i+1] = F_ring[i+1,i] = 0.994
            F_diag[i,i+1] = F_diag[i+1,i] = 0.986
            F_full[i,i+1] = F_full[i+1,i] = 0.988

    # rung for top qubit 15 → bottom qubit 31 (skipped by loop above)
    F_ring[15, 31] = F_ring[31, 15] = .995
    F_diag[15, 31] = F_diag[31, 15] = .987
    F_full[15, 31] = F_full[31, 15] = .991
    # backward diagonal 15→30 for full (forward 15→32 is out of bounds)
    F_full[15, 30] = F_full[30, 15] = 0.994

    if wraparound:
        F_ring[0, n//2-1]  = F_ring[n//2-1, 0]  = 0.996
        F_diag[0, n//2-1]  = F_diag[n//2-1, 0]  = 0.993
        F_full[0, n//2-1]  = F_full[n//2-1, 0]  = 0.993
        F_ring[n//2, n-1]  = F_ring[n-1, n//2]  = 0.994
        F_diag[n//2, n-1]  = F_diag[n-1, n//2]  = 0.986
        F_full[n//2, n-1]  = F_full[n-1, n//2]  = 0.989
        F_diag[n//2, n-1]  = F_diag[n-1, n//2]  = 0.994
        F_full[n//2, n-1]  = F_full[n-1, n//2]  = 0.975
        F_full[0, n-1]     = F_full[n-1, 0]     = 0.994

    # ── Pentagon topology (36 qubits: top row 0-17, bottom row 18-35) ───────────
    # 9 house-shaped modules (5q, 7-edge): 5 UP + 4 DOWN, alternating.
    # All modules have the same fidelity set {.989,.980,.973,.968,.968,.962,.960}.
    # DOWN is a true left-right mirror of UP.
    # Top row: period-4 [.960, .962, .962, .960] so each module gets one of each.
    # Bottom row: within-module = .968, inter-module junction = .962.
    F_pent=np.zeros((n,n))
    pentagon = {(0,1):.989,(1,0):.989,#left roof
                (3,4):.98,(4,3):.98,#base
                (1,2):.973,(2,1):.973,#right roof
                (0,3):.968,(3,0):.968,#left wall
                (2,4):.962,(4,2):.962,#right wall
                (1,3):.968,(3,1):.968,#left diagonal
                (1,4):.96,(4,1):.96} #right diagonal
    modules = [0,1,2,16,17],[19,18,17,3,2],[3,4,5,19,20],[22,21,20,6,5],[6,7,8,22,23],[25,24,23,9,8],[9,10,11,25,26],[28,27,26,12,11],[12,13,14,28,29],[31,30,29,15,14]

    for module in modules:
        for edge in pentagon:
            F_pent[module[edge[0]],module[edge[1]]]=pentagon[(edge[0],edge[1])]
    if wraparound:
        #todo
        pass

    cm_ring = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_ring[i,j]      > 0])
    cm_diag = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_diag[i,j]      > 0])
    cm_full = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_full[i,j]      > 0])
    cm_pent = CouplingMap([[i,j] for i in range(n) for j in range(n) if F_pent[i,j]      > 0])
    return [
        ("square_ring",      cm_ring, F_ring),
        ("square_ring_diag", cm_diag, F_diag),
        ("square_ring_full", cm_full, F_full),
        ("pentagon_ring",    cm_pent, F_pent)
    ]

# ── Configs ───────────────────────────────────────────────────────────────────
configs = [
    ("Standard SABRE",  dict(mode="lightsabre", aggression=0)),
    ("Standard MIRAGE", dict(mode="lightsabre", aggression=2)),
    ("FASST",           dict(mode="lightsabre", aggression=0,  edge_cost_weight=0.5)),
    ("FINESSE",         dict(mode="lightsabre", aggression=2,  fidelity_mirror=True, edge_cost_weight=0.5)),
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
def run_circuits(circuit_list, seed_list, label, out_path=None, wraparound=False, basis_gate='sqrt_iswap', devices=None):
    """
    circuit_list: list of (name, QuantumCircuit | qasm_filename_str)
    seed_list:    list of integer seeds to run (e.g. [0] for a single-seed parallel job)
    Writes rows to out_path as they complete (append mode, safe to run concurrently
    on disjoint seed_lists provided each job uses a distinct out_path).
    Returns a DataFrame.
    """
    if out_path is None:
        out_path = f"results_{label}.csv"
    fieldnames = ["suite", "device", "circuit", "config", "seed", "wraparound", "swaps", "depth", "lf_cost"]

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

            for cfg_name, kwargs in configs:
                kw = dict(kwargs)
                kw["basis_gate"] = basis_gate
                if cfg_name in FIDELITY_CONFIGS:
                    kw["fidelity_matrix"] = F
                for seed in seed_list:
                    rng = np.random.default_rng(seed + 10_000)
                    initial_cur = rng.permutation(n_phys).tolist()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        routed, _, _ = route(
                            copy.deepcopy(dag_phys), cm,
                            seed=seed, initial_cur=initial_cur, **kw,
                        )
                    row = dict(
                        suite=label, device=dev_name, circuit=circ_name,
                        config=cfg_name, seed=seed, wraparound=wraparound,
                        swaps=swap_count(routed),
                        depth=routed.depth(),
                        lf_cost=circuit_lf_cost(routed, F, basis_gate=basis_gate),
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
    ("qaoa_n25_p3",    make_qaoa(25, 3)),
    ("random_n32_d200",random_circuit(32, 200, max_operands=2, seed=42)),
]

if __name__ == "__main__":
    # ── Run ───────────────────────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",       action="store_true", help="Smoke test: 1 seed, 1 circuit per suite")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick sanity check: 3 small circuits, 3 seeds")
    parser.add_argument("--medium",     action="store_true",
                        help="Medium check: quick circuits + qft_n18 + qaoa_n20_p2 + random_n20_d100, 3 seeds")
    parser.add_argument("--suite",      choices=["redqueen", "ext_small", "ext_large", "all"], default="all",
                        help="Which suite to run (default: all)")
    parser.add_argument("--output",     default=None,
                        help="Output filename (without .csv). Default: results_{suite}. "
                             "Only valid when running a single suite.")
    parser.add_argument("--circuit",    default=None,
                        help="Run only this circuit (e.g. ising_n26). Must be used with --suite.")
    parser.add_argument("--seeds",      type=int, default=None,
                        help="Total number of seeds (default: per-suite value)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Run only this single seed. Output defaults to Results/<suite>_s<N>.csv. "
                             "Combine per-seed files with --merge.")
    parser.add_argument("--paper",      action="store_true",
                        help="Paper benchmark: 10 MQT circuits, 5 seeds, both wraparound variants")
    parser.add_argument("--stress",     action="store_true",
                        help="Stress benchmark: 6 medium QASMBench circuits, 5 seeds, both wraparound variants")
    parser.add_argument("--wraparound", action="store_true",
                        help="Add wraparound endpoint edges to the topology")
    parser.add_argument("--ibm",     action="store_true",
                        help="Run paper circuits on FakeTorontoV2 calibration topology")
    parser.add_argument("--merge",       action="store_true",
                        help="Merge per-seed CSVs (Results/paper_s*.csv) into Results/paper.csv and exit")
    parser.add_argument("--topology",   default="all",
                        help="Comma-separated topology names to run, or 'all' (default). "
                             "Options: square_ring, square_ring_diag, square_ring_full, pentagon_ring")
    parser.add_argument("--qasm",       default=None, metavar="FILE",
                        help="Run a single QASM file through all configs and topologies. "
                             "Circuit name defaults to the filename stem.")
    parser.add_argument("--compare",    action="store_true",
                        help="Compare 3 FASST/FINESSE variants on quick circuits to isolate config changes")
    args = parser.parse_args()

    if args.qasm:
        import os
        from qiskit import QuantumCircuit
        name = os.path.splitext(os.path.basename(args.qasm))[0]
        qc = QuantumCircuit.from_qasm_file(args.qasm)
        circuits = [(name, qc)]
        n_seeds = args.seeds if args.seeds is not None else 20
        seed_list = [args.seed] if args.seed is not None else list(range(n_seeds))
        out_path = f"{args.output}.csv" if args.output else f"Results/{name}.csv"
        for wrap in [False, True]:
            tag = "wrap" if wrap else "no-wrap"
            print(f"=== {name} ({n_seeds} seeds, {tag}) ===")
            run_circuits(circuits, seed_list=seed_list, label=name,
                         out_path=out_path, wraparound=wrap)
        df = pd.read_csv(out_path)
        print(df.groupby(["device", "config"])[["swaps", "depth", "lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.merge:
        import glob
        files = sorted(glob.glob("Results/paper_s*.csv"))
        if not files:
            print("No per-seed files found matching Results/paper_s*.csv")
            import sys; sys.exit(1)
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df.sort_values(["device","circuit","config","seed","wraparound"]).reset_index(drop=True)
        df.to_csv("Results/paper.csv", index=False)
        print(f"Merged {len(files)} files → Results/paper.csv ({len(df)} rows)")
        print(df.groupby(["device","config"])[["swaps","depth","lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.compare:
        if args.ibm:
            devs = build_ibm_topologies()
            basis_gate = 'cx'
            out_file = "compare_ibm.csv"
        else:
            devs = build_topology(wraparound=False)
            basis_gate = 'sqrt_iswap'
            out_file = "compare_check.csv"
        quick_circuits = [
            ("adder_n10",      fetch_qasmbench("adder_n10",      size="small")),
            ("multiplier_n15", fetch_qasmbench("multiplier_n15", size="medium")),
            ("bv_n19",         fetch_qasmbench("bv_n19",         size="medium")),
        ]
        compare_configs = [
            ("SABRE",           dict(mode="lightsabre", aggression=0)),
            ("MIRAGE",          dict(mode="lightsabre", aggression=2)),
            ("FINESSE b=.60",   dict(mode="lightsabre", aggression=2, fidelity_mirror=True, edge_cost_weight=0.5, fidelity_blend=.60)),
            ("FINESSE b=0.55",  dict(mode="lightsabre", aggression=2, fidelity_mirror=True, edge_cost_weight=0.5, fidelity_blend=0.55)),
            ("FINESSE b=0.5",  dict(mode="lightsabre", aggression=2, fidelity_mirror=True, edge_cost_weight=0.5, fidelity_blend=0.5)),
            ("FINESSE b=0.45",   dict(mode="lightsabre", aggression=2, fidelity_mirror=True, edge_cost_weight=0.5, fidelity_blend=0.45)),
        ]
        FIDELITY_COMPARE = {n for n, _ in compare_configs if "FINESSE" in n}
        configs[:] = compare_configs
        FIDELITY_CONFIGS.clear(); FIDELITY_CONFIGS.update(FIDELITY_COMPARE)
        n_seeds = args.seeds if args.seeds is not None else 5
        df = run_circuits(quick_circuits, seed_list=list(range(n_seeds)), label="compare",
                          out_path=out_file, wraparound=False, basis_gate=basis_gate, devices=devs)
        print(df.groupby(["device", "config"])[["swaps", "depth", "lf_cost"]].mean().round(3))
        import sys; sys.exit(0)

    if args.ibm:
        all_circuits = build_paper_circuits()
        if args.circuit:
            all_circuits = [(n, c) for n, c in all_circuits if n == args.circuit]
            if not all_circuits:
                parser.error(f"Circuit '{args.circuit}' not in paper suite. "
                             f"Options: {[n for n,_ in build_paper_circuits()]}")
        n_seeds = args.seeds if args.seeds is not None else 20
        seed_list = [args.seed] if args.seed is not None else list(range(n_seeds))
        out_path = f"{args.output}.csv" if args.output else (
            f"Results/ibm_s{args.seed}.csv" if args.seed is not None else "Results/ibm.csv"
        )
        for backend_name, cm, F in build_ibm_topologies():
            n_phys = cm.size()
            circuits = [(name, qc) for name, qc in all_circuits if qc.num_qubits <= n_phys]
            print(f"=== IBM {backend_name} ({n_phys}q, {len(circuits)} circuits, seeds={seed_list}) ===")
            run_circuits(circuits, seed_list=seed_list, label="ibm",
                         out_path=out_path, wraparound=False, basis_gate='cx',
                         devices=[(backend_name, cm, F)])
        df = pd.read_csv(out_path)
        ran = {n for n, _ in all_circuits}
        df = df[df["circuit"].isin(ran)]
        print(df.groupby(["device", "config"])[["swaps", "depth", "lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.paper:
        paper_circuits = build_paper_circuits()
        if args.circuit:
            paper_circuits = [(n, c) for n, c in paper_circuits if n == args.circuit]
            if not paper_circuits:
                parser.error(f"Circuit '{args.circuit}' not in paper suite. "
                             f"Options: {[n for n,_ in build_paper_circuits()]}")
        n_seeds = args.seeds if args.seeds is not None else 20
        if args.seed is not None:
            seed_list = [args.seed]
            default_out = f"Results/paper_s{args.seed}.csv"
        else:
            seed_list = list(range(n_seeds))
            default_out = "Results/paper.csv"
        out_path = f"{args.output}.csv" if args.output else default_out
        for wrap in [False, True]:
            devs = build_topology(wraparound=wrap)
            tag = "wrap" if wrap else "no-wrap"
            print(f"=== PAPER ({len(paper_circuits)} circuits, seeds={seed_list}, {tag}) ===")
            run_circuits(paper_circuits, seed_list=seed_list, label="paper",
                         out_path=out_path, wraparound=wrap, devices=devs)
        df = pd.read_csv(out_path)
        ran = {n for n, _ in paper_circuits}
        df = df[df["circuit"].isin(ran)]
        print(df.groupby(["device", "config"])[["swaps", "depth", "lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.stress:
        stress_circuits = build_stress_circuits()
        if args.circuit:
            stress_circuits = [(n, c) for n, c in stress_circuits if n == args.circuit]
            if not stress_circuits:
                parser.error(f"Circuit '{args.circuit}' not in stress suite. "
                             f"Options: {[n for n,_ in build_stress_circuits()]}")
        n_seeds = args.seeds if args.seeds is not None else 5
        if args.seed is not None:
            seed_list = [args.seed]
            default_out = f"Results/stress_s{args.seed}.csv"
        else:
            seed_list = list(range(n_seeds))
            default_out = "Results/stress.csv"
        out_path = f"{args.output}.csv" if args.output else default_out
        for wrap in [False, True]:
            devs = build_topology(wraparound=wrap)
            tag = "wrap" if wrap else "no-wrap"
            print(f"=== STRESS ({len(stress_circuits)} circuits, seeds={seed_list}, {tag}) ===")
            run_circuits(stress_circuits, seed_list=seed_list, label="stress",
                         out_path=out_path, wraparound=wrap, devices=devs)
        import pandas as pd
        df = pd.read_csv(out_path)
        ran = {n for n, _ in stress_circuits}
        df = df[df["circuit"].isin(ran)]
        print(df.groupby(["device", "config"])[["swaps", "depth", "lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.quick:
        devs = build_topology(wraparound=args.wraparound)
        quick_circuits = [
            ("adder_n10",      fetch_qasmbench("adder_n10",      size="small")),
            ("multiplier_n15", fetch_qasmbench("multiplier_n15", size="medium")),
            ("bv_n19",         fetch_qasmbench("bv_n19",         size="medium")),
        ]
        print(f"=== QUICK CHECK (3 circuits, 3 seeds{', wraparound' if args.wraparound else ''}) ===")
        quick_out = f"{args.output}.csv" if args.output else "quick_check.csv"
        df = run_circuits(quick_circuits, seed_list=list(range(3)), label="quick",
                          out_path=quick_out, wraparound=args.wraparound, devices=devs)
        print(df.groupby(["device","config"])[["swaps","depth","lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.medium:
        devs = build_topology(wraparound=args.wraparound)
        medium_circuits = [
            ("adder_n10",      fetch_qasmbench("adder_n10",      size="small")),
            ("multiplier_n15", fetch_qasmbench("multiplier_n15", size="medium")),
            ("bv_n19",         fetch_qasmbench("bv_n19",         size="medium")),
            ("qft_n18",        fetch_qasmbench("qft_n18",        size="medium")),
            ("qaoa_n20_p2",    make_qaoa(20, 2)),
            ("random_n20_d100",random_circuit(20, 100, max_operands=2, seed=42)),
        ]
        print(f"=== MEDIUM CHECK (6 circuits, 3 seeds{', wraparound' if args.wraparound else ''}) ===")
        medium_out = f"{args.output}.csv" if args.output else "medium_check.csv"
        df = run_circuits(medium_circuits, seed_list=list(range(3)), label="medium",
                          out_path=medium_out, wraparound=args.wraparound, devices=devs)
        print(df.groupby(["device","config"])[["swaps","depth","lf_cost"]].mean().round(2))
        import sys; sys.exit(0)

    if args.output and args.suite == "all":
        parser.error("--output can only be used with a specific --suite, not --suite all")
    if args.circuit and args.suite == "all":
        parser.error("--circuit requires a specific --suite")

    all_devices = build_topology(wraparound=args.wraparound)
    if args.topology == "all":
        devs = all_devices
    else:
        requested = {t.strip() for t in args.topology.split(",")}
        devs = [(name, cm, F) for name, cm, F in all_devices if name in requested]
        if not devs:
            parser.error(f"No matching topologies found. Available: {[d[0] for d in all_devices]}")
    if args.wraparound:
        print("Topology: wraparound enabled")
    print(f"Running on topologies: {[d[0] for d in devs]}")

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
        df = run_circuits(circuits_for_suite, seed_list=list(range(n_seeds)), label=name,
                          out_path=out_path, wraparound=args.wraparound, devices=devs)
        print(df.groupby(["device","config"])[["swaps","depth","lf_cost"]].mean().round(2))
