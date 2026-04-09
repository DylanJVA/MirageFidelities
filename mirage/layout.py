# mirage/layout.py

from __future__ import annotations

import numpy as np

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.basepasses import AnalysisPass


class FidelityLayout(AnalysisPass):
    """
    Layout pass that assigns logical qubits to the n physical qubits
    with the highest average neighbor fidelity.

    Each physical qubit p is scored by:
        s_p = mean(F[p, q] for q in neighbors(p))

    The n logical qubits are assigned to the n highest-scoring physical
    qubits in descending score order.

    This pass sets property_set["layout"] and is intended to be followed
    by FullAncillaAllocation, EnlargeWithAncilla, and ApplyLayout, matching
    the standard Qiskit layout pass interface.

    Note: this greedy strategy does not guarantee that the selected qubits
    form a well-connected subgraph. When high-scoring qubits are
    topologically scattered, the router may need additional SWAPs to
    connect them, potentially offsetting the fidelity benefit of the
    placement.

    Args:
        coupling_map:    Device connectivity.
        fidelity_matrix: Tuple-of-tuples of shape (n_qubits, n_qubits)
                         where F[i][j] is the fidelity of sqrt(iSWAP)
                         on physical link (i,j).
                         Pass as tuple(map(tuple, array)) for Qiskit
                         MetaPass hashability.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        fidelity_matrix: tuple,
    ):
        super().__init__()
        self.coupling_map = coupling_map
        self.fidelity_matrix = fidelity_matrix  # tuple-of-tuples

    def run(self, dag: DAGCircuit) -> None:
        """
        Compute and set property_set["layout"].
        Does not modify the DAG.
        """
        F = np.array(self.fidelity_matrix)
        n_virtual  = dag.num_qubits()
        n_physical = self.coupling_map.size()

        # Score each physical qubit by average neighbor fidelity
        scores = np.zeros(n_physical)
        for p in range(n_physical):
            neighbors = list(self.coupling_map.neighbors(p))
            if neighbors:
                scores[p] = np.mean([F[p, nb] for nb in neighbors])

        # Select top n_virtual physical qubits by score
        best_physical = np.argsort(scores)[::-1][:n_virtual]

        # Build layout: logical qubit i -> physical qubit best_physical[i]
        layout = Layout()
        for virt_idx, phys_idx in enumerate(dag.qubits):
            layout[phys_idx] = int(best_physical[virt_idx])

        self.property_set["layout"] = layout