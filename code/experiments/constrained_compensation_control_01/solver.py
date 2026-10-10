"""Issue-27 label-free, fixed-objective nonnegative compensation algebra.

The supplied factor and ridge are frozen Issue-25 inputs. No model, labels,
dataset, tuning outputs, or task gradients are accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear, nnls
from threadpoolctl import threadpool_limits

KKT_TOLERANCE = 1e-7
GAP_RELATIVE_TOLERANCE = 1e-8
GAP_OBJECTIVE_FLOOR = 1e-12
TARGET_RANKS = (8, 4, 2)
METHODS = ("C1-exact", "C2-exact")


class NumericalFailure(RuntimeError):
    """The frozen numerical certificate failed; scientific tolerances stay fixed."""


def _finite(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite {name}")
    return result


def _inputs(Z: Any, sigma: Any, keep_indices: Sequence[int], ridge: float):
    factor = _finite(Z, "factor")
    source = _finite(sigma, "source coefficients")
    if factor.ndim != 2 or source.ndim != 1 or factor.shape[0] < 1 or factor.shape[1] != source.size:
        raise ValueError("factor [examples, coordinates]/source coefficient shape mismatch")
    if np.any(source < 0):
        raise ValueError("source spectral coefficients must be nonnegative")
    raw = np.asarray(keep_indices)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size < 1 or raw.size >= source.size:
        raise ValueError("keep indices must be a nonempty nontrivial integer vector")
    keep = np.asarray(raw, dtype=np.int64)
    if np.any(keep < 0) or np.any(keep >= source.size) or len(np.unique(keep)) != len(keep):
        raise ValueError("duplicate or out-of-range keep indices")
    keep = np.sort(keep)
    absolute_ridge = float(ridge)
    if not np.isfinite(absolute_ridge) or absolute_ridge <= 0:
        raise ValueError("supplied absolute ridge must be positive and finite")
    mask = np.ones(source.size, dtype=bool)
    mask[keep] = False
    return factor, source, keep, np.flatnonzero(mask), absolute_ridge


def augmented_problem(Z: Any, sigma: Any, keep_indices: Sequence[int], *, ridge: float):
    """Return original-units augmented D,y without changing supplied inputs."""
    factor, source, keep, prune, absolute_ridge = _inputs(Z, sigma, keep_indices, ridge)
    n = len(factor)
    Zs = factor[:, keep]
    s = source[keep]
    b = factor[:, prune] @ source[prune]
    D = np.vstack((Zs / np.sqrt(n), np.sqrt(absolute_ridge) * np.eye(len(keep))))
    y = np.concatenate(((b + Zs @ s) / np.sqrt(n), np.sqrt(absolute_ridge) * s))
    if not np.isfinite(D).all() or not np.isfinite(y).all():
        raise ValueError("non-finite augmented problem")
    return D, y


def evaluate_objectives(Z: Any, sigma: Any, keep_indices: Sequence[int], coefficients: Any,
                        *, ridge: float, layer_slices: Mapping[str, slice] | None = None) -> dict[str, float]:
    """Evaluate one feasible full vector under common R and optional block R.

    The ridge penalty applies only to retained adjustments, exactly as in the
    protocol. Removed source coefficients are fixed at zero in every vector.
    """
    factor, source, keep, prune, absolute_ridge = _inputs(Z, sigma, keep_indices, ridge)
    candidate = _finite(coefficients, "candidate coefficients")
    if candidate.shape != source.shape or np.any(candidate < 0) or np.any(candidate[prune] != 0):
        raise ValueError("candidate must be full, feasible and exactly zero on pruned coordinates")
    delta = candidate - source
    residual = factor @ delta
    Q = float(np.dot(residual, residual) / (2 * len(factor)))
    penalty = float(0.5 * absolute_ridge * np.dot(delta[keep], delta[keep]))
    common_R = Q + penalty
    values = {"Q": Q, "common_R": common_R, "adjustment_norm": float(np.linalg.norm(delta[keep]))}
    if layer_slices is not None:
        _validate_layer_slices(layer_slices, source.size)
        own_Q = sum(float(np.dot(factor[:, sl] @ delta[sl], factor[:, sl] @ delta[sl]))
                    for sl in layer_slices.values()) / (2 * len(factor))
        values["layerwise_R"] = float(own_Q + penalty)
    if not np.isfinite(list(values.values())).all():
        raise ValueError("non-finite objective")
    return values


def certify_solution(Z: Any, sigma: Any, keep_indices: Sequence[int], c_keep: Any,
                     *, ridge: float) -> dict[str, Any]:
    """Independently certify c; exact-zero bounds define the dual multipliers.

    g = A c-h is evaluated via its equivalent residual form to avoid subtracting
    large near-equal vectors. We also record its direct equation discrepancy.
    The primal-dual gap is in original empirical objective units.
    """
    factor, source, keep, prune, absolute_ridge = _inputs(Z, sigma, keep_indices, ridge)
    c = _finite(c_keep, "retained coefficients")
    if c.shape != keep.shape:
        raise ValueError("retained coefficient shape mismatch")
    n = len(factor)
    Zs = factor[:, keep]
    s = source[keep]
    b = factor[:, prune] @ source[prune]
    x = c - s
    residual = Zs @ x - b
    Q = float(np.dot(residual, residual) / (2 * n))
    R = float(Q + 0.5 * absolute_ridge * np.dot(x, x))
    A = Zs.T @ Zs / n + absolute_ridge * np.eye(len(keep))
    h = A @ s + Zs.T @ b / n
    g = Zs.T @ residual / n + absolute_ridge * x
    active = c == 0.0
    free = ~active
    mu = np.where(active, np.maximum(g, 0.0), 0.0)
    coefficient_scale = max(float(np.max(np.abs(c))), float(np.max(np.abs(s))), 1e-30)
    gradient_scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(A @ c))), 1e-30)
    primal = float(np.max(np.maximum(-c, 0.0)))
    dual = float(np.max(np.maximum(-g[active], 0.0))) if active.any() else 0.0
    stationarity = float(np.max(np.abs(g[free]))) if free.any() else 0.0
    complementarity = float(np.max(np.abs(c * g)))
    primal_scaled = primal / coefficient_scale
    dual_scaled = dual / gradient_scale
    stationarity_scaled = stationarity / gradient_scale
    complementarity_scaled = complementarity / max(coefficient_scale * gradient_scale, 1e-30)
    scaled_kkt = max(primal_scaled, dual_scaled, stationarity_scaled, complementarity_scaled)
    dual_error = g - mu
    # Positive ridge makes A strictly positive definite; no inverse is formed.
    with threadpool_limits(limits=1):
        solved_error = cho_solve(cho_factor(A, lower=True, check_finite=True), dual_error)
    gap = float(0.5 * np.dot(dual_error, solved_error) + np.dot(mu, c))
    gap_tolerance = GAP_RELATIVE_TOLERANCE * max(R, GAP_OBJECTIVE_FLOOR)
    finite = np.isfinite([R, Q, gap, scaled_kkt, gradient_scale, coefficient_scale]).all()
    passed = bool(finite and primal == 0.0 and gap >= 0.0 and
                  scaled_kkt <= KKT_TOLERANCE and gap <= gap_tolerance)
    return {
        "passed": passed, "R": R, "Q": Q, "scaled_kkt_residual": scaled_kkt,
        "primal_nonnegativity_residual": primal, "dual_at_bounds_residual": dual,
        "free_stationarity_residual": stationarity, "complementarity_residual": complementarity,
        "primal_scaled": primal_scaled, "dual_scaled": dual_scaled,
        "stationarity_scaled": stationarity_scaled, "complementarity_scaled": complementarity_scaled,
        "gradient_scale": gradient_scale, "coefficient_scale": coefficient_scale,
        "primal_dual_gap": gap, "gap_tolerance": gap_tolerance,
        "kkt_tolerance": KKT_TOLERANCE, "gap_relative_tolerance": GAP_RELATIVE_TOLERANCE,
        "gap_objective_floor": GAP_OBJECTIVE_FLOOR, "active_count": int(active.sum()),
        "free_count": int(free.sum()), "active_indices": np.flatnonzero(active).tolist(),
        "multiplier_rule": "mu_j=max(g_j,0) exactly where c_j==0; mu_j=0 elsewhere; no snapping",
        "gradient_evaluation": "Zs.T@(Zs@(c-s)-b)/N+ridge*(c-s), equivalent to A@c-h",
        "normal_equation_gradient_discrepancy": float(np.max(np.abs((A @ c - h) - g))),
        "unconstrained_equation_residual": float(np.linalg.norm(g)),
    }


@dataclass(frozen=True)
class NonnegativeSolution:
    c_keep: np.ndarray
    delta_keep: np.ndarray
    ridge: float
    R: float
    Q: float
    certificate: dict[str, Any]
    solver_metadata: dict[str, Any]
    solve_seconds: float


def _bounded_roundoff(bounded) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize only solver-declared bounds at floating-point roundoff scale.

    BVLS sometimes returns a declared zero bound at +/-1e-17. Materially
    nonzero values are never snapped, and the entire certificate is recomputed
    after this explicitly recorded representation normalization.
    """
    values = np.array(bounded.x, dtype=np.float64, copy=True)
    threshold = 64 * np.finfo(np.float64).eps * max(float(np.max(np.abs(values))), 1e-30)
    normalize = (np.asarray(bounded.active_mask) == -1) & (np.abs(values) <= threshold) & (values != 0)
    indices = np.flatnonzero(normalize)
    record = {"rule": "solver active_mask==-1 and abs(c)<=64*eps*max(abs(c),1e-30)",
              "roundoff_threshold": float(threshold), "indices": indices.tolist(),
              "original_values": values[indices].tolist(), "full_certificate_recomputed": True}
    values[indices] = 0.0
    return values, record


def solve_nonnegative(Z: Any, sigma: Any, keep_indices: Sequence[int], *, ridge: float,
                      independent_check: bool = True) -> NonnegativeSolution:
    """Solve augmented NNLS, retrying the same objective only on numerical failure."""
    start = perf_counter()
    factor, source, keep, _, absolute_ridge = _inputs(Z, sigma, keep_indices, ridge)
    D, y = augmented_problem(factor, source, keep, ridge=absolute_ridge)
    # One common scalar affects all rows and columns identically. It changes
    # neither the estimator nor the certificate's original objective units.
    scale = float(1.0 / max(float(np.max(np.abs(D))), float(np.max(np.abs(y))), 1e-30))
    Ds, ys = D * scale, y * scale
    attempts: list[dict[str, Any]] = []
    selected = None
    bounded = None
    bounded_values = None
    bounded_roundoff = None
    bounded_certificate = None
    with threadpool_limits(limits=1):
        try:
            c, augmented_residual = nnls(Ds, ys, maxiter=max(1000, 30 * len(keep)))
            cert = certify_solution(factor, source, keep, c, ridge=absolute_ridge)
            attempts.append({"solver": "scipy.optimize.nnls", "certificate_passed": cert["passed"],
                             "scaled_augmented_residual_l2": float(augmented_residual)})
            if cert["passed"]:
                selected = (c, cert, "scipy.optimize.nnls")
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            attempts.append({"solver": "scipy.optimize.nnls", "error": str(exc), "certificate_passed": False})
        if selected is None or independent_check:
            try:
                bounded = lsq_linear(Ds, ys, bounds=(0.0, np.inf), method="bvls", tol=1e-13, max_iter=10000)
                bounded_values, bounded_roundoff = _bounded_roundoff(bounded)
                bounded_certificate = certify_solution(factor, source, keep, bounded_values, ridge=absolute_ridge)
                record = {"solver": "scipy.optimize.lsq_linear(bvls)",
                          "role": "numerical_retry" if selected is None else "independent_check",
                          "success": bool(bounded.success), "status": int(bounded.status),
                          "message": str(bounded.message), "iterations": int(bounded.nit),
                          "certificate_passed": bounded_certificate["passed"],
                          "bound_roundoff_normalization": bounded_roundoff}
                attempts.append(record)
                if selected is None and bounded_certificate["passed"]:
                    selected = (bounded_values, bounded_certificate, "scipy.optimize.lsq_linear(bvls)")
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                attempts.append({"solver": "scipy.optimize.lsq_linear(bvls)", "error": str(exc),
                                 "certificate_passed": False})
    if selected is None:
        raise NumericalFailure(f"Both standard numerical routes failed frozen certificate: {attempts}")
    if independent_check and (bounded_certificate is None or not bounded_certificate["passed"]):
        raise NumericalFailure(f"Independent bounded LS certificate failed: {attempts}")
    c, cert, chosen_solver = selected
    metadata = {
        "solver": chosen_solver, "scipy_version": scipy.__version__, "dtype": "float64",
        "formulation": "augmented nonnegative ridge least squares", "global_scalar_rescaling": scale,
        "nnls_maxiter": max(1000, 30 * len(keep)), "bounded_method": "bvls", "bounded_tol": 1e-13,
        "bounded_max_iter": 10000, "attempts": attempts, "numerical_retry": chosen_solver != "scipy.optimize.nnls",
        "independent_check_requested": bool(independent_check), "blas_thread_limit": 1,
    }
    if bounded is not None and bounded_certificate is not None:
        metadata["independent_certificate"] = bounded_certificate
        metadata["independent_coefficient_relative_l2"] = float(
            np.linalg.norm(c - bounded_values) / max(float(np.linalg.norm(c)), 1e-30))
        metadata["independent_R_difference"] = float(bounded_certificate["R"] - cert["R"])
    return NonnegativeSolution(c_keep=np.array(c, dtype=np.float64, copy=True),
                               delta_keep=np.array(c - source[keep], dtype=np.float64, copy=True),
                               ridge=absolute_ridge, R=cert["R"], Q=cert["Q"],
                               certificate=cert, solver_metadata=metadata, solve_seconds=perf_counter() - start)


def _validate_layer_slices(layer_slices: Mapping[str, slice], dimension: int) -> None:
    start = 0
    for name in sorted(layer_slices):
        sl = layer_slices[name]
        if not isinstance(sl, slice) or sl.start != start or sl.stop != start + 32 or sl.step not in (None, 1):
            raise ValueError("layer map must cover contiguous sorted source-rank32 coordinates")
        start = int(sl.stop)
    if not layer_slices or start != dimension:
        raise ValueError("layer map must cover all source coefficients")


@dataclass(frozen=True)
class ConstrainedCondition:
    method: str
    target_rank: int
    coefficients: np.ndarray
    delta: np.ndarray
    keep_indices: np.ndarray
    prune_indices: np.ndarray
    ridge: float
    own_R: float
    common_R: float
    Q_cal: float
    adjustment_norm: float
    certificate: dict[str, Any]
    layer_solve_diagnostics: dict[str, dict[str, Any]]
    solve_seconds: float

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.method, "target_rank": self.target_rank, "ridge": self.ridge,
            "global_matched_ridge": self.ridge, "coefficient_constraint": "nonnegative",
            "coefficient_dimension": int(self.coefficients.size), "kept_coefficients": int(self.keep_indices.size),
            "pruned_coefficients": int(self.prune_indices.size), "own_R": self.own_R,
            "own_objective": "sum of independent layer R" if self.method == "C1-exact" else "global R",
            "common_R": self.common_R, "Q_cal": self.Q_cal, "adjustment_norm": self.adjustment_norm,
            "certificate": self.certificate, "layer_solve_diagnostics": self.layer_solve_diagnostics,
            "solve_seconds": self.solve_seconds, "quadratic_target": "calibration KL(source || candidate)",
        }


def build_conditions(Z: Any, sigma: Any, layer_slices: Mapping[str, slice], target_rank: int,
                     *, ridge: float) -> dict[str, ConstrainedCondition]:
    """Construct C1 and C2 with one identical caller-supplied global ridge."""
    if target_rank not in TARGET_RANKS:
        raise ValueError("frozen target ranks are 8,4,2")
    source = _finite(sigma, "source coefficients")
    _validate_layer_slices(layer_slices, source.size)
    keep = np.concatenate([np.arange(layer_slices[name].start, layer_slices[name].start + target_rank,
                                    dtype=np.int64) for name in sorted(layer_slices)])
    factor, source, keep, prune, absolute_ridge = _inputs(Z, source, keep, ridge)
    conditions = {}
    for method in METHODS:
        start = perf_counter()
        coefficients = np.zeros_like(source)
        diagnostics = {}
        if method == "C1-exact":
            solutions = []
            for name in sorted(layer_slices):
                sl = layer_slices[name]
                solved = solve_nonnegative(factor[:, sl], source[sl], np.arange(target_rank), ridge=absolute_ridge)
                coefficients[sl.start:sl.start + target_rank] = solved.c_keep
                diagnostics[name] = {"ridge": solved.ridge, "R": solved.R, "Q": solved.Q,
                                     "certificate": solved.certificate, "solver_metadata": solved.solver_metadata,
                                     "solve_seconds": solved.solve_seconds}
                solutions.append(solved)
            own_R = sum(solution.R for solution in solutions)
            gaps = [solution.certificate["primal_dual_gap"] for solution in solutions]
            own_gap = float(sum(gaps))
            own_gap_tolerance = GAP_RELATIVE_TOLERANCE * max(own_R, GAP_OBJECTIVE_FLOOR)
            if own_gap > own_gap_tolerance:
                raise NumericalFailure("C1 summed own-objective gap failed frozen aggregate target")
            certificate = {
                "passed": all(solution.certificate["passed"] for solution in solutions),
                "objective_scope": "sum of independent layer objectives; not global R",
                "scaled_kkt_residual": max(solution.certificate["scaled_kkt_residual"] for solution in solutions),
                "primal_dual_gap": own_gap, "gap_tolerance": own_gap_tolerance,
                "max_layer_primal_dual_gap": max(gaps), "R": float(own_R),
                "layer_count": len(solutions), "ridge": absolute_ridge,
                "kkt_tolerance": KKT_TOLERANCE, "gap_relative_tolerance": GAP_RELATIVE_TOLERANCE,
                "gap_objective_floor": GAP_OBJECTIVE_FLOOR,
                "all_layer_gap_targets_passed": all(solution.certificate["primal_dual_gap"] <=
                                                     solution.certificate["gap_tolerance"] for solution in solutions),
            }
        else:
            solved = solve_nonnegative(factor, source, keep, ridge=absolute_ridge)
            coefficients[keep] = solved.c_keep
            own_R = solved.R
            certificate = dict(solved.certificate, objective_scope="global R")
            diagnostics["global"] = {"ridge": solved.ridge, "R": solved.R, "Q": solved.Q,
                                     "certificate": solved.certificate, "solver_metadata": solved.solver_metadata,
                                     "solve_seconds": solved.solve_seconds}
        objectives = evaluate_objectives(factor, source, keep, coefficients, ridge=absolute_ridge,
                                         layer_slices=layer_slices)
        conditions[method] = ConstrainedCondition(method=method, target_rank=target_rank,
            coefficients=coefficients, delta=coefficients - source, keep_indices=keep.copy(), prune_indices=prune.copy(),
            ridge=absolute_ridge, own_R=float(own_R), common_R=objectives["common_R"], Q_cal=objectives["Q"],
            adjustment_norm=objectives["adjustment_norm"], certificate=certificate,
            layer_solve_diagnostics=diagnostics, solve_seconds=perf_counter() - start)
    return conditions
