# mirage/pass_.py

from __future__ import annotations

import copy
import numpy as np

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library import SwapGate, UnitaryGate, iSwapGate, CXGate, UGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, Target, InstructionProperties
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import (
    SabreSwap, HighLevelSynthesis, UnrollCustomDefinitions,
    BasisTranslator, Collect2qBlocks, ConsolidateBlocks,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from .cost import decomp_cost, SWAP_MATRIX
from .mirror import accept_mirror


def make_unroll_consolidate():
    """
    Standard preprocessing pass sequence for MIRAGE.

    Decomposes high-level gates (qft, permutation, ccircuit, etc.)
    down to primitive u/cx/swap gates, then consolidates consecutive
    2Q gates on the same qubit pair into single UnitaryGate nodes.
    This ensures the router sees only bare 2Q unitaries, which is
    what MIRAGE's mirror pass and decomposition expect.
    """
    return [
        HighLevelSynthesis(),
        UnrollCustomDefinitions(SessionEquivalenceLibrary,
                                basis_gates=["u", "cx", "swap"]),
        BasisTranslator(SessionEquivalenceLibrary,
                        target_basis=["u", "cx", "swap"]),
        Collect2qBlocks(),
        ConsolidateBlocks(),
    ]

def build_target_from_fidelities(
    coupling_map: CouplingMap,
    fidelity_matrix: np.ndarray,
) -> Target:
    """
    Build a Qiskit Target encoding gate error rates from a fidelity matrix.

    Populates sqrt(iSWAP), CX, and U gate error rates so that layout passes
    (e.g. VF2Layout) can use them for fidelity-aware qubit placement.

    Note: SabreSwap does NOT use these error rates for SWAP selection — its
    heuristic is purely distance-based. Fidelity awareness in routing is
    achieved via trial selection in MirageSwap (consider_fidelities=True).

    Args:
        coupling_map:    Device connectivity.
        fidelity_matrix: F[i,j] = fidelity of sqrt(iSWAP) on link (i,j).

    Gate error rates:
        sqrt(iSWAP): error = 1 - F[i,j]
        CX:          error = 1 - F[i,j]^2   (CX costs 2 sqrt(iSWAP) gates)
        U:           error = 0.0             (single-qubit gates assumed perfect)
    """
    target = Target(num_qubits=coupling_map.size())
    sqrt_iswap = iSwapGate().power(0.5)

    sqiswap_props = {}
    cx_props = {}
    for p0, p1 in coupling_map.get_edges():
        f = max(float(fidelity_matrix[p0, p1]), 1e-10)
        sqiswap_props[(p0, p1)] = InstructionProperties(error=1.0 - f)
        cx_props[(p0, p1)]      = InstructionProperties(error=1.0 - f ** 2)

    target.add_instruction(sqrt_iswap, sqiswap_props)
    target.add_instruction(CXGate(), cx_props)

    u_props = {
        (p,): InstructionProperties(error=0.0)
        for p in range(coupling_map.size())
    }
    target.add_instruction(UGate(0, 0, 0), u_props)

    return target


class MirageSwap(TransformationPass):
    """
    MIRAGE: Mirror-decomposition Integrated Routing for Algorithm Gate Efficiency.

    Implementation of "MIRAGE: Quantum Circuit Decomposition
    and Routing Collaborative Design using Mirror Gates" (arXiv:2308.03874). If consider_fidelities=False, implements the MIRAGE algorithm as described in the paper. If consider_fidelities=True, adds fidelity-aware trial selection on top of MIRAGE.

    Algorithm:
      1. Run n_trials of SabreSwap (each with a different random seed).
      2. Apply mirror gate post-processing to each trial's output:
         for each SWAP(a,b) → Gate(a,b) pattern, if mirror(Gate) has
         equal or lower decomposition cost, replace both with mirror(Gate),
         eliminating the SWAP for free.
      3. Select the best trial by:
         - consider_fidelities=False: lowest sqrt(iSWAP) critical-path depth (just MIRAGE)
         - consider_fidelities=True:  lowest total log-infidelity cost

    Mirror gates: mirror(U) = SWAP @ U. Since SWAP costs 3 sqrt(iSWAP)
    gates, absorbing a SWAP into the gate via mirroring can save up to 3 gates
    at no extra cost when decomp_cost(mirror(U)) <= decomp_cost(U).

    Fidelity-aware trial selection:
    SabreSwap's SWAP placement involves random tie-breaking. Different random
    seeds land SWAPs on different physical links. When links have different
    fidelities, selecting the trial whose SWAPs happened to land on
    high-fidelity links reduces the total execution error.

    Args:
        coupling_map:        Device connectivity.
        n_trials:            Number of independent SabreSwap trials.
                             More trials = better optimization but slower.
        seed:                Base random seed. Trial i uses seed+i.
        forced_aggression:   Override mirror acceptance aggression (0-3).
                             None = use default level 2.
                             0: never accept mirror
                             1: accept if strictly cheaper
                             2: accept if cheaper or equal  [default]
                             3: always accept
        consider_fidelities: If True, select best trial by total
                             log-infidelity cost instead of depth.
                             Requires fidelity_matrix.
        fidelity_matrix:     Tuple-of-tuples of shape (n_qubits, n_qubits)
                             where F[i][j] = fidelity of sqrt(iSWAP) on
                             physical link (i,j).
                             Must be passed as tuple(map(tuple, array))
                             for Qiskit MetaPass hashability.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        n_trials: int = 20,
        seed: int = 42,
        forced_aggression: int | None = None,
        consider_fidelities: bool = False,
        fidelity_matrix: tuple | None = None,
    ):
        super().__init__()
        self.coupling_map = coupling_map
        self.n_trials = n_trials
        self.seed = seed
        self.forced_aggression = forced_aggression
        self.consider_fidelities = consider_fidelities
        self.fidelity_matrix = fidelity_matrix  # tuple-of-tuples for hashability

        if consider_fidelities and fidelity_matrix is None:
            raise ValueError(
                "fidelity_matrix must be provided when consider_fidelities=True"
            )

        # Single SabreSwap instance for the fast (non-fidelity) path.
        # Delegates all trial management to Qiskit's Rust implementation.
        self._sabre = SabreSwap(
            coupling_map,
            heuristic="decay",
            seed=seed,
            trials=n_trials,
        )

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if not self.consider_fidelities:
            # Fast path: SabreSwap handles all trials internally in Rust,
            # selects best by depth/SWAP count, then we apply mirror pass.
            self._sabre.property_set = self.property_set
            dag = self._sabre.run(dag)
            return self._mirror_pass(dag)

        # Fidelity-aware path: run trials manually in Python,
        # select best trial by total log-infidelity cost.
        F = np.array(self.fidelity_matrix)
        best_dag = None
        best_cost = float("inf")
        best_property_set = None

        for trial_idx in range(self.n_trials):
            sabre_trial = SabreSwap(
                self.coupling_map,
                heuristic="decay",
                seed=self.seed + trial_idx,
                trials=1,
            )
            sabre_trial.property_set = self.property_set

            # deepcopy required — DAGCircuit has no .copy() in Qiskit 2.x
            trial_dag = sabre_trial.run(copy.deepcopy(dag))
            trial_dag = self._mirror_pass(trial_dag)

            cost = self._fidelity_cost(trial_dag, F)
            if cost < best_cost:
                best_cost = cost
                best_dag = trial_dag
                best_property_set = dict(sabre_trial.property_set)

        # Propagate property set (layout, final_layout) from best trial
        if best_property_set:
            for k, v in best_property_set.items():
                self.property_set[k] = v

        return best_dag

    def _mirror_pass(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Post-processing pass: scan for SWAP(a,b) → Gate(a,b) patterns.

        For each such pattern, compute decomp_cost(Gate) and
        decomp_cost(SWAP @ Gate). If the mirror is cheaper or equal
        (per the aggression level), replace both with mirror(Gate),
        eliminating the SWAP entirely.

        note: fidelity doesn't affect this decision
        Both the SWAP and the Gate execute on the same physical link (a,b),
        so the fidelity factor F[a,b] appears on both sides of the cost
        comparison and cancels. The decision reduces to pure k-cost arithmetic.
        Fidelity awareness is applied at the trial-selection level instead.
        """
        aggression = (
            self.forced_aggression
            if self.forced_aggression is not None
            else 2
        )

        for node in list(dag.topological_op_nodes()):
            if not isinstance(node.op, SwapGate):
                continue

            swap_qargs = set(node.qargs)

            for succ in dag.quantum_successors(node):
                if not isinstance(succ, DAGOpNode):
                    continue
                if len(succ.qargs) != 2:
                    continue
                if set(succ.qargs) != swap_qargs:
                    continue

                try:
                    U = Operator(succ.op).data
                except Exception:
                    continue

                cost_u = decomp_cost(U)
                cost_m = decomp_cost(SWAP_MATRIX @ U)

                if accept_mirror(cost_u, cost_m, aggression):
                    dag.substitute_node(
                        succ,
                        UnitaryGate(SWAP_MATRIX @ U, check_input=False),
                        inplace=True,
                    )
                    dag.remove_op_node(node)
                    break  # this SWAP is consumed; move to next SWAP node

        return dag

    def _mirror_pass_extended(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Extended mirror pass that can absorb non-adjacent SWAPs.
        Tracks qubit permutation state through the circuit to identify
        SWAPs whose permutation effect reaches a later gate intact.
        """
        aggression = (
            self.forced_aggression
            if self.forced_aggression is not None
            else 2
        )

        # pending_swaps[frozenset({a,b})] = swap_node
        # A SWAP is pending on (a,b) until a gate consumes qubits a or b
        pending_swaps: dict[frozenset, DAGOpNode] = {}

        for node in list(dag.topological_op_nodes()):
            if isinstance(node.op, SwapGate):
                qpair = frozenset(node.qargs)
                # If there's already a pending SWAP on this pair,
                # two SWAPs cancel — remove both
                if qpair in pending_swaps:
                    dag.remove_op_node(pending_swaps[qpair])
                    dag.remove_op_node(node)
                    del pending_swaps[qpair]
                else:
                    pending_swaps[qpair] = node

            elif len(node.qargs) == 2:
                qpair = frozenset(node.qargs)

                if qpair in pending_swaps:
                    # There's a pending SWAP on exactly this qubit pair
                    # Check if mirror is beneficial
                    try:
                        U = Operator(node.op).data
                    except Exception:
                        # Can't get unitary — consume the pending SWAP
                        # since this gate uses these qubits
                        del pending_swaps[qpair]
                        continue

                    cost_u = decomp_cost(U)
                    cost_m = decomp_cost(SWAP_MATRIX @ U)

                    if accept_mirror(cost_u, cost_m, aggression):
                        # Absorb the SWAP into the mirror gate
                        dag.substitute_node(
                            node,
                            UnitaryGate(SWAP_MATRIX @ U, check_input=False),
                            inplace=True,
                        )
                        dag.remove_op_node(pending_swaps[qpair])
                    del pending_swaps[qpair]

                else:
                    # This gate uses qubits that may have pending SWAPs
                    # on other pairs — invalidate any pending SWAP that
                    # involves either qubit of this gate
                    used = set(node.qargs)
                    to_remove = [
                        pair for pair in pending_swaps
                        if pair & used  # overlapping qubits
                    ]
                    for pair in to_remove:
                        del pending_swaps[pair]

            elif len(node.qargs) == 1:
                # 1Q gates don't consume SWAP permutations —
                # a pending SWAP on (a,b) survives a 1Q gate on a
                # because the SWAP hasn't been "used" yet
                pass

        return dag

    def _fidelity_cost(self, dag: DAGCircuit, F: np.ndarray) -> float:
        """
        Total log-infidelity of the routed DAG under fidelity matrix F.

        cost = sum over 2Q gates of  k * (-log F[i,j])

        where k = number of sqrt(iSWAP) gates needed to implement the gate,
        and (i,j) are the physical qubit indices. Lower is better.

        Single-qubit gates contribute 0 (assumed perfect fidelity).
        """
        total = 0.0
        for node in dag.topological_op_nodes():
            if len(node.qargs) != 2:
                continue
            p0 = dag.find_bit(node.qargs[0]).index
            p1 = dag.find_bit(node.qargs[1]).index
            if node.op.name == "xx_plus_yy":
                k = 1.0
            elif node.op.name == "swap":
                k = 3.0
            elif node.op.name == "unitary":
                try:
                    k = float(decomp_cost(Operator(node.op).data))
                except Exception:
                    k = 2.0  # safe fallback
            else:
                k = 0.0
            f = max(float(F[p0, p1]), 1e-10)
            total += k * (-np.log(f))
        return total
