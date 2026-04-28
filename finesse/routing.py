# finesse/routing.py
#
# SABRE / LightSABRE / MIRAGE / FINESSE routing.
#
# Implements Algorithm 1 from Li et al. (SABRE, ASPLOS 2019) with the
# LightSABRE modifications from Zou et al. (arXiv:2409.08368):
#   - Unnormalized basic heuristic term (LightSABRE `mode`)
#   - BFS extended set with predecessor-readiness tracking (both modes)
#   - Decay multiplier max(δ[p0], δ[p1]) with periodic reset
#   - Release valve: cycle detection → backtrack → Dijkstra escape
#
# H — one heuristic, used for both SWAP selection and mirror acceptance:
#
#   No fidelity:   H = max(δ) · (Σ_F D_hop + W·avg_E D_hop)
#   With fidelity: H = max(δ) · (Σ_F D_fid + W·avg_E D_fid)
#
# Mirror acceptance (MIRAGE, aggression > 0):
#   No fidelity:   compare H(current layout) vs H(permuted layout)
#   With fidelity: compare k(U)·lf + H(current) vs k(U')·lf + H(permuted)
#                  where lf = -log F[edge] and k(.) is the Weyl-chamber
#                  decomposition cost in the native 2Q basis.
#
# Correctness invariants:
#   - Every op-node in the input DAG is emitted exactly once.
#   - cur[orig] = current physical qubit of logical qubit orig;
#     rev[phys] = orig such that cur[orig] = phys.  Always consistent.
#   - A SWAP(p0,p1) updates cur/rev via swap_positions BEFORE any
#     mirror check or successor flush, so qargs are always evaluated
#     against the current layout.
#   - Mirror absorption: emit(SWAP·U) and mark_executed(nid) — no
#     separate SWAP emitted; layout already updated.
#   - Release valve backtrack: del ops[idx] in reverse index order so
#     earlier indices are still valid when reached.

from __future__ import annotations

import copy
import heapq
import numpy as np

from .mirror import circuit_lf_cost
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate, UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from .mirror import decomp_cost, SWAP_MATRIX, accept_mirror

# ---------------------------------------------------------------------------
# Hyperparameters (match Qiskit's .with_decay(0.001, 5) and lookahead(0.5, 20))
# ---------------------------------------------------------------------------
EXTENDED_SET_WEIGHT    = 0.5   # W: lookahead weight relative to front layer
EXTENDED_SET_SIZE      = 20    # max gates in extended set E
DECAY_RATE             = 0.001 # δ increment per SWAP on a qubit
DECAY_RESET            = 5     # consecutive SWAPs before resetting all δ to 1
ATTEMPT_LIMIT_FACTOR   = 10    # valve fires after (factor × n_qubits) SWAPs without progress
                               # matches Qiskit's Heuristic(attempt_limit=10 * num_dag_qubits)
SCORE_EPSILON          = 1e-10 # tie-breaking threshold for heuristic score comparison
DIST_FID_SWAP_WEIGHT   = 3.0   # edge weight multiplier in D_fid: 1.0 = single-gate lf, 3.0 = SWAP cost
FIDELITY_FLOOR         = 1e-10 # clamp F[i,j] away from zero before taking -log


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_adj(coupling_map: CouplingMap, n: int) -> list[list[int]]:
    """Undirected adjacency list — coupling_map edges may be directed."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in coupling_map.get_edges():
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _dijkstra(adj: list[list[int]], src: int) -> tuple[list[float], list[int]]:
    """Dijkstra from src on an undirected adjacency list.

    Returns (cost, prev) where cost[v] is the shortest distance from src
    and prev[v] is the predecessor of v on that path (-1 if unvisited).
    """
    n = len(adj)
    cost = [float('inf')] * n
    prev = [-1] * n
    cost[src] = 0
    heap = [(0, src)]
    while heap:
        c, u = heapq.heappop(heap)
        if c > cost[u]:
            continue
        for v in adj[u]:
            if cost[u] + 1 < cost[v]:
                cost[v] = cost[u] + 1
                prev[v] = u
                heapq.heappush(heap, (cost[v], v))
    return cost, prev


def _build_dist(coupling_map: CouplingMap) -> np.ndarray:
    """D[i][j] = shortest-path hop-count distance between physical qubits i and j."""
    n = coupling_map.size()
    adj = _build_adj(coupling_map, n)
    d = np.full((n, n), np.inf)
    for src in range(n):
        cost, _ = _dijkstra(adj, src)
        d[src] = cost
    return d


def _build_dist_fid(coupling_map: CouplingMap, L_raw: np.ndarray) -> np.ndarray:
    """D_fid[i][j] = min Σ(-log F[e]) over paths i→j (Dijkstra with -log F weights).

    When fidelity is uniform, D_fid reduces to a scalar multiple of the hop-count
    distance.  When edges differ in fidelity, longer paths through reliable edges
    can beat shorter paths through noisy ones.
    """
    n = coupling_map.size()
    adj_w: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v in coupling_map.get_edges():
        w = DIST_FID_SWAP_WEIGHT * float(L_raw[u, v])
        adj_w[u].append((v, w))
        adj_w[v].append((u, w))
    d = np.full((n, n), np.inf)
    for src in range(n):
        dist_src = [float('inf')] * n
        dist_src[src] = 0.0
        heap: list[tuple[float, int]] = [(0.0, src)]
        while heap:
            c, u = heapq.heappop(heap)
            if c > dist_src[u]:
                continue
            for v, w in adj_w[u]:
                nc = c + w
                if nc < dist_src[v]:
                    dist_src[v] = nc
                    heapq.heappush(heap, (nc, v))
        d[src] = dist_src
    return d


def _dijkstra_path(coupling_map: CouplingMap, src: int, dst: int,
                   L_raw: np.ndarray | None = None) -> list[int]:
    """Shortest path from src to dst.

    If L_raw is provided uses fidelity-weighted edges (same weights as D_fid),
    otherwise falls back to hop-count (unit weights).
    """
    n = coupling_map.size()
    # Build weighted adjacency list once
    adj_w: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b in coupling_map.get_edges():
        w = DIST_FID_SWAP_WEIGHT * float(L_raw[a, b]) if L_raw is not None else 1.0
        adj_w[a].append((b, w))
        adj_w[b].append((a, w))

    cost = [float('inf')] * n
    prev = [-1] * n
    cost[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        c, u = heapq.heappop(heap)
        if c > cost[u]:
            continue
        for v, w in adj_w[u]:
            nc = cost[u] + w
            if nc < cost[v]:
                cost[v] = nc
                prev[v] = u
                heapq.heappush(heap, (nc, v))
    path = []
    u = dst
    while u != -1:
        path.append(u)
        u = prev[u]
    return list(reversed(path))


def _orig_phys(dag: DAGCircuit, qubit) -> int:
    return dag.find_bit(qubit).index


def _build_deps(dag: DAGCircuit, *, reverse: bool = False):
    """
    Build predecessor counts and successor id lists for all op nodes.
    Returns (pred_count, successors, nodes).

    reverse=True: process gates back-to-front.  A node is "ready" when all
    its forward-successors have been processed.
    """
    pred_count: dict[int, int] = {}
    successors: dict[int, list[int]] = {}
    nodes = {}

    for node in dag.topological_op_nodes():
        nid = node._node_id
        nodes[nid] = node
        if reverse:
            pred_count[nid] = sum(
                1 for s in dag.successors(node) if isinstance(s, DAGOpNode)
            )
        else:
            pred_count[nid] = sum(
                1 for p in dag.predecessors(node) if isinstance(p, DAGOpNode)
            )
        successors[nid] = []

    for node in dag.topological_op_nodes():
        nid = node._node_id
        if reverse:
            for pred in dag.predecessors(node):
                if isinstance(pred, DAGOpNode):
                    successors[nid].append(pred._node_id)
        else:
            for succ in dag.successors(node):
                if isinstance(succ, DAGOpNode):
                    successors[nid].append(succ._node_id)

    return pred_count, successors, nodes


# ---------------------------------------------------------------------------
# Layout warmup pass  (bidirectional SABRE initialisation)
# ---------------------------------------------------------------------------

def _layout_pass(
    dag: DAGCircuit,
    coupling_map: CouplingMap,
    initial_cur: list[int],
    *,
    reverse: bool = False,
    seed: int = 0,
) -> list[int]:
    """One SABRE routing pass for layout warmup; no ops emitted."""
    _, _, final_cur = route(
        dag, coupling_map,
        seed=seed,
        mode='sabre',
        aggression=0,
        valve=False,
        bidir_passes=0,
        use_decay=True,
        initial_cur=list(initial_cur),
        reverse=reverse,
        emit_ops=False,
    )
    return final_cur


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

def route(
    dag: DAGCircuit,
    coupling_map: CouplingMap,
    aggression: int = 2,
    seed: int = 0,
    mode: str = 'lightsabre',
    valve: bool | None = None,
    bidir_passes: int | None = None,
    basis_gate: str = 'sqrt_iswap',
    fidelity_matrix: np.ndarray | None = None,
    fidelity_mirror: bool = True,
    edge_cost_weight: float = 0.0,
    fidelity_blend: float = .5,
    use_decay: bool = False,
    initial_cur: list[int] | None = None,
    n_trials: int = 1,
    emit_ops: bool = True,
    reverse: bool = False,
) -> tuple[DAGCircuit | None, int, list[int]]:
    """
    SABRE / LightSABRE / MIRAGE / FINESSE routing.

    Implements Algorithm 1 (Li et al. ASPLOS 2019) with LightSABRE modifications
    (Zou et al. arXiv:2409.08368) and the MIRAGE/FINESSE intermediate layer.

    Args:
        dag:              Circuit to route (DAGCircuit with physical layout applied).
                          Use apply_trivial_layout() or Qiskit's ApplyLayout first.
        coupling_map:     Device connectivity graph.
        aggression:       Mirror acceptance level (0–3, default 2):
                            0 = never mirror
                            1 = mirror only if strictly cheaper
                            2 = mirror if cheaper or equal  [default]
                            3 = always mirror
        seed:             Random seed for tie-breaking.
        mode:             Heuristic variant:
                            'lightsabre' (default) — unnormalized basic term
                                H = max(δ[p0],δ[p1]) · (∑_F D  +  W·(1/|E|)·∑_E D)
                            'sabre' — paper §4, normalised by |F|
                                H = max(δ[p0],δ[p1]) · ((1/|F|)·∑_F D  +  W·(1/|E|)·∑_E D)
                          Extended set E uses BFS with predecessor-readiness tracking
                          in both modes (matches Qiskit's populate_extended_set).
        valve:            Enable release valve (backtrack + Dijkstra escape when
                          cycling is detected). Defaults to True for 'lightsabre',
                          False for 'sabre'.
        bidir_passes:     Bidirectional layout warmup rounds before the final pass.
                          Defaults to 0 (disabled).
        basis_gate:       Native 2Q gate for Weyl decomposition cost.
                          Supported: 'sqrt_iswap', 'cx'. Defaults to 'sqrt_iswap'.
        fidelity_matrix:  Optional (n, n) array where F[i,j] is the 2Q gate fidelity
                          on link (i,j). When provided, H uses D_fid (lf-weighted
                          Dijkstra) for routing distances instead of D_hop.
        fidelity_mirror:  Whether fidelity enters the intermediate mirror layer
                          (default True). When True and fidelity_matrix is provided,
                          mirror acceptance compares k_U·lf + H_fid(current) vs
                          k_U'·lf + H_fid(permuted). When False, mirror acceptance
                          uses hop-count layout scores regardless of fidelity_matrix.
                          Ignored when aggression=0 or fidelity_matrix=None.
        initial_cur:      Optional initial logical→physical mapping.
        n_trials:         Number of independent routing trials (default 1). Each
                          trial uses seed+i. Best trial selected by total
                          -log-fidelity cost (when fidelity_matrix provided)
                          or gate depth otherwise.

          Ablation configurations (use BenchmarkConfig for convenience):
            1. SABRE:                mode='sabre',       aggression=0
            2. LightSABRE:           mode='lightsabre',  aggression=0
            3. MIRAGE (SABRE):       mode='sabre',       aggression=2
            4. MIRAGE (LightSABRE):  mode='lightsabre',  aggression=2
            5. SABRE + fidelity:     mode='sabre',       aggression=0,  fidelity_matrix=F
            6. LightSABRE + fid:     mode='lightsabre',  aggression=0,  fidelity_matrix=F
            7. MIRAGE + fid (SABRE): mode='sabre',       aggression=2,  fidelity_matrix=F, fidelity_mirror=False
            8. MIRAGE + fid (LS):    mode='lightsabre',  aggression=2,  fidelity_matrix=F, fidelity_mirror=False
            9. FINESSE:              mode='lightsabre',  aggression=2,  fidelity_matrix=F, fidelity_mirror=True

    Returns:
        (routed DAGCircuit, valve_fires, final_cur)
    """
    if mode not in ('sabre', 'lightsabre'):
        raise ValueError(f"mode must be 'sabre' or 'lightsabre', got {mode!r}")

    # Multi-trial post-selection: run n_trials independent single-trial routes
    # and return the best by -log-fidelity cost (fidelity_matrix provided) or depth.
    if n_trials > 1:
        import copy
        best_dag, best_vf, best_cur = None, 0, []
        best_score = float('inf')
        for i in range(n_trials):
            t_dag, t_vf, t_cur = route(
                copy.deepcopy(dag), coupling_map,
                aggression=aggression, seed=seed + i, mode=mode,
                valve=valve, bidir_passes=bidir_passes, basis_gate=basis_gate,
                fidelity_matrix=fidelity_matrix, fidelity_mirror=fidelity_mirror,
                edge_cost_weight=edge_cost_weight, fidelity_blend=fidelity_blend,
                use_decay=use_decay, initial_cur=initial_cur, n_trials=1,
                emit_ops=emit_ops, reverse=reverse,
            )
            score = (
                circuit_lf_cost(t_dag, fidelity_matrix, basis_gate)
                if fidelity_matrix is not None
                else t_dag.depth()
            )
            if score < best_score:
                best_score, best_dag, best_vf, best_cur = score, t_dag, t_vf, t_cur
        return best_dag, best_vf, best_cur

    if valve is None:
        valve = (mode == 'lightsabre')
    if bidir_passes is None:
        bidir_passes = 0
    if not emit_ops:
        valve = False  # valve backtracks via ops indices; incompatible with emit_ops=False

    rng  = np.random.default_rng(seed)
    n    = dag.num_qubits()
    dist = _build_dist(coupling_map)
    adj  = _build_adj(coupling_map, n)

    # Log-infidelity matrices:
    # L_raw[i,j] = -log F[i,j]  (raw log-infidelity; all H terms use this unit)
    L_raw: np.ndarray | None = None
    if fidelity_matrix is not None:
        L_raw = -np.log(np.maximum(fidelity_matrix, FIDELITY_FLOOR))

    # dist_fid[i,j] = min-lf Dijkstra path cost from i to j.
    # Used in H_dist when fidelity is active (same lf units as L_raw).
    dist_fid: np.ndarray | None = None
    if L_raw is not None:
        d_fid = _build_dist_fid(coupling_map, L_raw)
        if fidelity_blend < 1.0:
            # Blend raw hop-count and lf-weighted distances without normalising.
            # Normalising squashes distances to [0,1], making them tiny relative
            # to the raw-lf edge_penalty on large/noisy backends (e.g. Washington
            # 127q), which causes the edge penalty to dominate and break routing.
            dist_fid = (1.0 - fidelity_blend) * dist + fidelity_blend * d_fid
        else:
            dist_fid = d_fid

    # ------------------------------------------------------------------
    # §IV.C / Fig. 5 — Reverse traversal for initial mapping
    # One bidir round = one forward pass + one backward pass on the reversed DAG.
    # The resulting layout is used as the starting point for the final forward pass.
    # Off by default (bidir_passes=0); enable with bidir_passes=1 or more.
    # ------------------------------------------------------------------
    if initial_cur is None:
        initial_cur = list(range(n))
    for _ in range(bidir_passes):
        initial_cur = _layout_pass(dag, coupling_map, initial_cur,
                                   reverse=False, seed=seed)
        initial_cur = _layout_pass(dag, coupling_map, initial_cur,
                                   reverse=True, seed=seed)

    # ------------------------------------------------------------------
    # Layout tracking — invariant maintained throughout §IV.B Algorithm 1
    #   cur[orig] = current physical qubit of logical qubit orig
    #   rev[phys] = orig  s.t.  cur[orig] = phys
    # ------------------------------------------------------------------
    cur = list(initial_cur)
    rev = [0] * n
    for orig, phys in enumerate(cur):
        rev[phys] = orig

    def swap_positions(p0: int, p1: int) -> None:
        o0, o1 = rev[p0], rev[p1]
        cur[o0], cur[o1] = p1, p0
        rev[p0], rev[p1] = o1, o0

    def cur_phys_of(node) -> tuple[int, ...]:
        return tuple(cur[_orig_phys(dag, q)] for q in node.qargs)

    # ------------------------------------------------------------------
    # Dependency tracking — pred_count[nid] = unexecuted predecessors of gate nid
    # ready = {nid : pred_count[nid] == 0} = the current front layer F ∪ 1Q gates
    # ------------------------------------------------------------------
    pred_count, successors, nodes = _build_deps(dag, reverse=reverse)
    executed: set[int] = set()
    ready: set[int] = {nid for nid, c in pred_count.items() if c == 0}

    def mark_executed(nid: int) -> None:
        executed.add(nid)
        ready.discard(nid)
        for sid in successors[nid]:
            pred_count[sid] -= 1
            if pred_count[sid] == 0:
                ready.add(sid)

    # ------------------------------------------------------------------
    # Output collection
    # ------------------------------------------------------------------
    ops: list[tuple] = []

    def emit(op, phys_qargs: tuple, clbits: tuple = ()) -> None:
        if emit_ops:
            ops.append((op, phys_qargs, clbits))

    def _clbit_indices(node) -> tuple:
        return tuple(dag.find_bit(c).index for c in node.cargs)

    # ------------------------------------------------------------------
    # H_dist sum over a gate set, using dist_fid (lf units).
    # Used for mirror acceptance: same H as SWAP selection, applied to current/permuted layout.
    # ------------------------------------------------------------------
    def _h_finesse_sum(gate_ids: list[int], ext_ids: list[int]) -> float:
        assert dist_fid is not None
        f_sum = sum(float(dist_fid[cur_phys_of(nodes[g])[0]][cur_phys_of(nodes[g])[1]])
                    for g in gate_ids)
        e_sum = (sum(float(dist_fid[cur_phys_of(nodes[g])[0]][cur_phys_of(nodes[g])[1]])
                     for g in ext_ids) / len(ext_ids)
                 if ext_ids else 0.0)
        return f_sum + EXTENDED_SET_WEIGHT * e_sum

    # §IV.B — Execute_gate_list  (Algorithm 1, lines 2–10)
    # Drain every gate that is currently executable:
    #   - 1Q gates (and barriers, measurements): always executable immediately.
    #   - 2Q gates: executable when their physical qubits are adjacent.
    #
    
    # Intermediate mirror layer (aggression > 0): when a 2Q gate is routable,
    # compare executing as U vs U' = SWAP·U using the same H as SWAP selection.
    # H is pure hop-count when no fidelity_matrix; fidelity-weighted otherwise.
    # If U' is accepted, emit U' and apply the implicit layout SWAP.
    # ------------------------------------------------------------------
    def flush_executable() -> None:
        changed = True
        while changed:
            changed = False
            for nid in list(ready):
                node = nodes[nid]
                nq = len(node.qargs)
                if nq != 2:
                    emit(node.op, cur_phys_of(node), _clbit_indices(node))
                    mark_executed(nid); changed = True
                elif cur_phys_of(node)[0] in adj[cur_phys_of(node)[1]]:
                    gps = cur_phys_of(node)
                    mirrored = False
                    U_mirror: np.ndarray | None = None

                    if aggression > 0:
                        gp0, gp1 = gps
                        # Use ALL other ready 2Q gates (not just non-adjacent ones).
                        # A mirror absorption implicitly swaps (gp0, gp1), which can
                        # un-adjacent a currently-ready gate that shares a qubit — a cost
                        # invisible if we only score stuck (non-adjacent) gates.
                        other_f = [
                            sid for sid in ready if sid != nid
                            and len(nodes[sid].qargs) == 2
                        ]
                        # Include successors of G itself in the extended set.
                        # An implicit SWAP changes positions of (gp0, gp1), so only
                        # G's successors see a routing-distance change.  Other front-
                        # layer gates cannot share qubits with G (qubit exclusivity),
                        # so their distances are unaffected by the swap.  Without G's
                        # successors in E, H_perm == H_cur always and mirror never fires.
                        E_mirror = extended_set(other_f + [nid])

                        if dist_fid is not None and fidelity_mirror:
                            # Fidelity-aware: compare k_U*lf + H_fid(cur) vs k_U'*lf + H_fid(perm).
                            #   k_U/k_U' use the native basis_gate decomp cost (Weyl-chamber,
                            #   basis-independent). On sqrt_iswap hardware, k_CX = k_{SWAP·CX} = 2,
                            #   so the criterion is H_perm ≤ H_cur (accept when routing doesn't worsen).
                            #   On CX hardware, k_CX=1 and k_{SWAP·CX}=2: accept when H_perm ≤ H_cur-lf.
                            #   lf   = -log F[gp0, gp1]  (raw, same units as dist_fid)
                            # H_fid uses D_fid (lf-weighted Dijkstra paths) over all other
                            # front-layer gates — the same H as SWAP selection.
                            try:
                                U_mat      = Operator(node.op).data
                                U_mir_mat  = SWAP_MATRIX @ U_mat
                                k_U        = float(decomp_cost(U_mat,      basis_gate))
                                k_Up       = float(decomp_cost(U_mir_mat,  basis_gate))
                                lf_gate    = float(L_raw[gp0, gp1])
                                H_cur      = _h_finesse_sum(other_f, E_mirror)
                                swap_positions(gp0, gp1)
                                H_perm     = _h_finesse_sum(other_f, E_mirror)
                                swap_positions(gp0, gp1)  # undo
                                cost_U     = k_U  * lf_gate + H_cur
                                cost_Up    = k_Up * lf_gate + H_perm
                                if accept_mirror(cost_U, cost_Up, aggression):
                                    mirrored = True
                                    U_mirror = U_mir_mat
                            except Exception:
                                pass
                        else:
                            # Hop-count: compare k_U + H(cur) vs k_U' + H(perm).
                            # Uses native basis_gate decomp cost (Weyl-chamber, basis-independent).
                            # On sqrt_iswap: k_CX = k_{SWAP·CX} = 2 → accept when H_perm ≤ H_cur.
                            # On CX:         k_CX=1, k_{SWAP·CX}=2 → accept when H_perm ≤ H_cur - 1.
                            def _layout_score() -> float:
                                F_sum = sum(
                                    dist[cur_phys_of(nodes[s])[0]][cur_phys_of(nodes[s])[1]]
                                    for s in other_f
                                )
                                E_sum = (
                                    sum(dist[cur_phys_of(nodes[s])[0]][cur_phys_of(nodes[s])[1]]
                                        for s in E_mirror) / len(E_mirror)
                                    if E_mirror else 0.0
                                )
                                return F_sum + EXTENDED_SET_WEIGHT * E_sum

                            try:
                                U_mat     = Operator(node.op).data
                                U_mir_mat = SWAP_MATRIX @ U_mat
                                k_U       = float(decomp_cost(U_mat,      basis_gate))
                                k_Up      = float(decomp_cost(U_mir_mat,  basis_gate))
                                score_cur  = k_U  + _layout_score()
                                swap_positions(gp0, gp1)
                                score_perm = k_Up + _layout_score()
                                swap_positions(gp0, gp1)          # undo
                                if accept_mirror(score_cur, score_perm, aggression):
                                    U_mirror = U_mir_mat
                                    mirrored = True
                            except Exception:
                                pass

                    if mirrored and U_mirror is not None:
                        emit(UnitaryGate(U_mirror, check_input=False),
                             (gp0, gp1), _clbit_indices(node))
                        swap_positions(gp0, gp1)                 # apply implicit SWAP
                    else:
                        emit(node.op, gps, _clbit_indices(node))
                    mark_executed(nid); changed = True

    def front_layer_2q() -> list[int]:
        """§IV.B — Front layer F: 2Q gates with all predecessors executed."""
        return [nid for nid in ready if len(nodes[nid].qargs) == 2]

    # §IV.B — SWAP_candidate_list: all coupling-map edges touching a qubit in F.
    # ------------------------------------------------------------------
    def obtain_swaps(F: list[int]) -> set[tuple[int, int]]:
        S: set[tuple[int, int]] = set()
        for gate in F:
            for q in nodes[gate].qargs:
                p = cur[_orig_phys(dag, q)]
                for nb in adj[p]:
                    S.add((min(p, nb), max(p, nb)))
        return S

    # §IV.C.2 — Extended set E (look-ahead ability and parallelism)
    # BFS from F with predecessor-readiness tracking: a gate is enqueued
    # only when all its unexecuted predecessors have been visited in the
    # current BFS traversal, propagating transitively through 1Q chains.
    # Capped at EXTENDED_SET_SIZE=20. Matches Qiskit's populate_extended_set.
    # ------------------------------------------------------------------
    def extended_set(F: list[int]) -> list[int]:
        to_visit = list(F)
        decremented: dict[int, int] = {}
        E: list[int] = []
        i = 0
        while i < len(to_visit) and len(E) < EXTENDED_SET_SIZE:
            gate = to_visit[i]
            for succ in successors[gate]:
                decremented[succ] = decremented.get(succ, 0) + 1
                pred_count[succ] -= 1
                if pred_count[succ] == 0:
                    to_visit.append(succ)
                    if len(nodes[succ].qargs) == 2:
                        E.append(succ)
            i += 1
        for succ, amt in decremented.items():
            pred_count[succ] += amt
        return E

    # §IV.D — Heuristic H  (one candidate SWAP scored per call)
    #
    # No fidelity:
    #   H = max(δ) · (Σ_F D_hop + W·avg_E D_hop)
    #
    # With fidelity:
    #   H = max(δ) · (Σ_F D_fid + W·avg_E D_fid)
    #
    # Mirror acceptance uses this same downstream H term, and when fidelity is
    # active it additionally weights the gate's own decomposition cost as
    # k(U)·(-log F_edge).
    #
    def heuristic_score(
        p0: int, p1: int,
        F: list[int], E: list[int],
        decay: np.ndarray,
    ) -> float:
        π  = cur_phys_of
        W  = EXTENDED_SET_WEIGHT

        # Select distance matrix: D_fid (lf units) when fidelity active, D_hop otherwise.
        _d = dist_fid if dist_fid is not None else dist

        swap_positions(p0, p1)  # simulate

        F_sum = sum(_d[π(nodes[g])[0]][π(nodes[g])[1]] for g in F)
        E_sum = sum(_d[π(nodes[g])[0]][π(nodes[g])[1]] for g in E) if E else 0.0

        # SABRE normalises the basic term by |F|; LightSABRE does not.
        basic = (F_sum / len(F)) if mode == 'sabre' else F_sum
        decay_factor = float(max(decay[p0], decay[p1])) if use_decay else 1.0
        edge_penalty=0
        #edge_penalty = (edge_cost_weight * DIST_FID_SWAP_WEIGHT * float(L_raw[p0, p1])
        #                if dist_fid is not None and edge_cost_weight > 0.0 else 0.0)
        H_dist = decay_factor * (basic + W * (E_sum / len(E) if E else 0.0) + edge_penalty)

        swap_positions(p0, p1)  # undo simulation
        return H_dist

    # §IV.B — SWAP* ← argmin_{SWAP ∈ S} H(SWAP, F, E, D, δ)  (Algorithm 1, line 14)
    # Ties broken uniformly at random (SCORE_EPSILON threshold).
    # ------------------------------------------------------------------
    def choose_swap(
        S: set[tuple[int, int]], F: list[int], E: list[int],
        decay: np.ndarray,
    ) -> tuple[int, int]:
        best_score = float('inf')
        best: list[tuple[int, int]] = []
        for (p0, p1) in S:
            h = heuristic_score(p0, p1, F, E, decay)
            if h < best_score - SCORE_EPSILON:
                best_score = h; best = [(p0, p1)]
            elif abs(h - best_score) < SCORE_EPSILON:
                best.append((p0, p1))
        return best[int(rng.integers(len(best)))]

    # §IV.B — Algorithm 1: SABRE main loop
    # Extensions active depending on configuration:
    #   mode='lightsabre'          → unnormalized F term in H (LightSABRE)
    #   valve=True                 → release valve on stall detection (LightSABRE)
    #   aggression > 0             → intermediate mirror layer at gate execution
    #   fidelity_matrix is not None → fidelity-weighted H in both SWAP selection
    #                                 and intermediate mirror layer
    # ------------------------------------------------------------------
    decay = np.ones(n)
    num_search_steps: int = 0   # counts consecutive SWAPs; only periodic reset
    flush_executable()          # execute initial front layer

    # swap_history tracks (p0, p1, ops_idx) for each SWAP since the last
    # executed gate — used by the release valve to backtrack.
    swap_history: list[tuple[int, int, int]] = []
    attempt_limit = ATTEMPT_LIMIT_FACTOR * n   # matches Qiskit's 10 * num_dag_qubits
    max_iter = dag.size() * n + 10_000
    valve_fires = 0

    for _ in range(max_iter):
        F = front_layer_2q()
        if not F:
            break

        # ---- LightSABRE — Release valve ------------------------------------
        # Stall detection: more than attempt_limit consecutive SWAPs without
        # executing any gate (attempt_limit = 10 × n_qubits, matching Qiskit).
        # Escape: backtrack all SWAPs to the last gate-execution checkpoint,
        # then force-route the nearest front-layer gate via Dijkstra
        # (SWAPs inserted from both path ends toward the midpoint).
        if valve and len(swap_history) >= attempt_limit:
            for bp0, bp1, idx in reversed(swap_history):
                swap_positions(bp0, bp1)
                del ops[idx]
            swap_history.clear()
            valve_fires += 1

            _vdist = dist_fid if dist_fid is not None else dist
            best_nid = min(F, key=lambda nid: _vdist[cur_phys_of(nodes[nid])[0]][cur_phys_of(nodes[nid])[1]])
            bp0, bp1 = cur_phys_of(nodes[best_nid])
            path = _dijkstra_path(coupling_map, bp0, bp1, L_raw=L_raw)
            mid = len(path) // 2
            for i in range(mid - 1):
                pa, pb = path[i], path[i + 1]
                swap_positions(pa, pb); emit(SwapGate(), (pa, pb))
            for i in range(len(path) - 1, mid, -1):
                pa, pb = path[i], path[i - 1]
                swap_positions(pa, pb); emit(SwapGate(), (pa, pb))

            flush_executable()
            continue
        # ----------------------------------------------------------------

        S      = obtain_swaps(F)
        if not S:
            break
        E      = extended_set(F)
        p0, p1 = choose_swap(S, F, E, decay)

        # ---- Apply SWAP and record for valve backtrack ----------------------
        # Mirror decisions are made at gate execution time in flush_executable
        # (the MIRAGE / FINESSE intermediate layer), not here.
        swap_positions(p0, p1)
        emit(SwapGate(), (p0, p1))
        swap_history.append((p0, p1, len(ops) - 1))

        # §IV.C.3 — Decay δ update
        # Increment δ for the two SWAP qubits; periodic reset every DECAY_RESET
        # steps.  Also reset whenever a gate executes (progress made).
        num_search_steps += 1
        if num_search_steps >= DECAY_RESET:
            decay[:] = 1.0
            num_search_steps = 0
        else:
            decay[p0] += DECAY_RATE
            decay[p1] += DECAY_RATE

        gates_before = len(executed)
        flush_executable()          # §IV.B — Execute_gate_list after SWAP
        if len(executed) > gates_before:
            swap_history.clear()
            decay[:] = 1.0          # reset δ on gate execution
            num_search_steps = 0    # reset counter so periodic reset realigns

    # Final flush
    flush_executable()

    # ------------------------------------------------------------------
    # Correctness fallback: if max_iter exhausted before all gates executed,
    # force-route remaining 2Q gates via Dijkstra.  This should never fire
    # in normal use — enable the valve or increase max_iter if it does.
    # ------------------------------------------------------------------
    fallback_gates = 0
    F = front_layer_2q()
    while F:
        fallback_gates += 1
        best_nid = min(
            F,
            key=lambda nid: dist[cur_phys_of(nodes[nid])[0]]
                                 [cur_phys_of(nodes[nid])[1]]
        )
        bp0, bp1 = cur_phys_of(nodes[best_nid])
        path = _dijkstra_path(coupling_map, bp0, bp1)
        mid = len(path) // 2
        for i in range(mid - 1):
            pa, pb = path[i], path[i + 1]
            swap_positions(pa, pb); emit(SwapGate(), (pa, pb))
        for i in range(len(path) - 1, mid, -1):
            pa, pb = path[i], path[i - 1]
            swap_positions(pa, pb); emit(SwapGate(), (pa, pb))
        flush_executable()
        F = front_layer_2q()

    if fallback_gates:
        import warnings
        warnings.warn(
            f"route: fallback Dijkstra routed {fallback_gates} gate(s) — "
            f"max_iter exhausted before routing completed. "
            f"SWAP count may be inflated. Consider increasing max_iter or enabling the valve.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Build output DAG
    # ------------------------------------------------------------------
    if not emit_ops:
        return None, valve_fires, list(cur)

    qc = QuantumCircuit(dag.num_qubits(), dag.num_clbits())
    for op, phys_qargs, clbits in ops:
        qc.append(op, list(phys_qargs), list(clbits))

    out_dag = circuit_to_dag(qc)

    return out_dag, valve_fires, list(cur)
