# finesse/layout.py

from __future__ import annotations

import numpy as np

from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.basepasses import AnalysisPass

from .mirror import decomp_cost
from .routing import route, _layout_pass


class FinesseLayout(AnalysisPass):
    """
    SABRE-style layout pass with fidelity-aware trial scoring.

    Runs n_trials routing trials, each starting from a random initial
    logical→physical assignment (optionally refined by bidirectional SABRE
    warmup). Scores each completed routing by total raw -log(f) cost:

        cost = Σ_{2Q ops} decomp_cost(op) · (-log F[p0, p1])
               + Σ_{SWAPs} 3 · (-log F[p0, p1])

    The layout from the lowest-cost trial is kept and written to
    property_set["layout"].

    Unlike FidelityLayout (which scores individual qubits by average
    neighbor fidelity without circuit context), FinesseLayout accounts
    for the circuit's actual interaction structure: high-interaction
    qubit pairs are pushed onto high-fidelity edges by the routing
    heuristic, and the trial score measures that placement directly.

    This pass sets property_set["layout"] and must be followed by
    FullAncillaAllocation, EnlargeWithAncilla, and ApplyLayout.

    Args:
        coupling_map:    Device connectivity.
        fidelity_matrix: F[i,j] = fidelity of the native 2Q gate on link (i,j).
                         Must be tuple-of-tuples (not ndarray) for Qiskit MetaPass
                         hashability. Convert with tuple(map(tuple, F)).
        n_trials:        Number of random-layout trials (default 10).
        seed:            Master RNG seed (default 0).
        aggression:      Mirror absorption level for routing trials (0–3, default 2).
        basis_gate:      Native 2Q gate for decomposition cost ('sqrt_iswap' | 'cx').
        bidir_passes:    Bidirectional warmup rounds per trial (default 1).
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        fidelity_matrix,
        n_trials: int = 10,
        seed: int = 0,
        aggression: int = 2,
        basis_gate: str = 'sqrt_iswap',
        bidir_passes: int = 1,
    ):
        super().__init__()
        self.coupling_map = coupling_map
        self.fidelity_matrix = fidelity_matrix
        self.n_trials = n_trials
        self.seed = seed
        self.aggression = aggression
        self.basis_gate = basis_gate
        self.bidir_passes = bidir_passes

    def run(self, dag: DAGCircuit) -> None:
        F = np.array(self.fidelity_matrix)
        L_raw = -np.log(np.maximum(F, 1e-10))
        n_virtual = dag.num_qubits()
        n_physical = self.coupling_map.size()
        rng = np.random.default_rng(self.seed)

        best_cost = float('inf')
        best_initial_cur: list[int] = list(range(n_virtual))

        for _ in range(self.n_trials):
            # Random initial layout: draw n_virtual physical qubits without replacement
            trial_seed = int(rng.integers(2**31))
            phys_order = rng.permutation(n_physical)
            initial_cur = list(phys_order[:n_virtual])

            # Bidirectional distance-based warmup to refine the random seed layout
            trial_rng_seed = int(rng.integers(2**31))
            for _ in range(self.bidir_passes):
                initial_cur = _layout_pass(
                    dag, self.coupling_map, initial_cur,
                    reverse=False, seed=trial_rng_seed,
                )
                initial_cur = _layout_pass(
                    dag, self.coupling_map, initial_cur,
                    reverse=True, seed=trial_rng_seed,
                )

            # Route with this initial layout (fidelity-aware heuristic active)
            routed_dag, _, _ = route(
                dag, self.coupling_map,
                aggression=self.aggression,
                seed=trial_seed,
                fidelity_matrix=F,
                basis_gate=self.basis_gate,
                initial_cur=initial_cur,
            )

            # Score: Σ decomp_cost(op) * (-log F[p0,p1]) over all 2Q ops in routed circuit
            cost = 0.0
            for node in routed_dag.topological_op_nodes():
                if len(node.qargs) != 2:
                    continue
                p0 = routed_dag.find_bit(node.qargs[0]).index
                p1 = routed_dag.find_bit(node.qargs[1]).index
                lf = float(L_raw[p0, p1])
                if isinstance(node.op, SwapGate):
                    cost += 3.0 * lf
                else:
                    try:
                        gate_cost = decomp_cost(Operator(node.op).data, self.basis_gate)
                    except Exception:
                        gate_cost = 1.0
                    cost += gate_cost * lf

            if cost < best_cost:
                best_cost = cost
                best_initial_cur = initial_cur

        # Write the best initial layout into property_set
        layout = Layout()
        for virt_idx, qubit in enumerate(dag.qubits):
            layout[qubit] = int(best_initial_cur[virt_idx])

        self.property_set["layout"] = layout
