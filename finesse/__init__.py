# finesse/__init__.py

# ── Routing ───────────────────────────────────────────────────────────────────
from .routing     import route
from .inline_pass import InlineMirageSwap

# ── Qiskit pipeline passes ────────────────────────────────────────────────────
from .layout import FinesseLayout
try:
    from .decompose import MirageDecompose
except ModuleNotFoundError:
    MirageDecompose = None

# ── Pipeline setup ────────────────────────────────────────────────────────────
from .benchmarks import make_unroll_consolidate, apply_trivial_layout

# ── Fidelity ──────────────────────────────────────────────────────────────────
from .fidelity import fidelity_matrix_from_backend, build_target_from_fidelities
from .mirror import circuit_lf_cost

# ── Benchmark utilities ───────────────────────────────────────────────────────
from .benchmarks import (
    make_tokyo, fetch_qasm, fetch_qasmbench,
    check_routing, evaluate_routing_checks,
    print_routing_check_report, run_correctness_suite, run_config_correctness_suite,
    run_clifford_correctness_suite, random_clifford_circuit,
    line_cm, grid_cm, prepare_dag, benchmark_mode, swap_count,
    strip_regs, permutation_correction_qc,
    available_redqueen_circuits, available_qasmbench_circuits,
    load_redqueen_circuits, load_qasmbench_circuits,
    BenchmarkConfig, run_benchmark,
)

__all__ = [
    "route",
    "InlineMirageSwap",
    "FinesseLayout",
    "MirageDecompose",
    "make_unroll_consolidate",
    "apply_trivial_layout",
    "fidelity_matrix_from_backend",
    "build_target_from_fidelities",
    "circuit_lf_cost",
    "make_tokyo",
    "fetch_qasm",
    "fetch_qasmbench",
    "check_routing",
    "evaluate_routing_checks",
    "print_routing_check_report",
    "run_correctness_suite",
    "run_config_correctness_suite",
    "run_clifford_correctness_suite",
    "random_clifford_circuit",
    "line_cm",
    "grid_cm",
    "prepare_dag",
    "benchmark_mode",
    "swap_count",
    "strip_regs",
    "permutation_correction_qc",
    "available_redqueen_circuits",
    "available_qasmbench_circuits",
    "load_redqueen_circuits",
    "load_qasmbench_circuits",
    "BenchmarkConfig",
    "run_benchmark",
]
