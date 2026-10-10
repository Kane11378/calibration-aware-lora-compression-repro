"""Label-free nested F/B/D Fisher geometry with the unchanged Issue-27 solver."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.constrained_compensation_control_01.solver import solve_nonnegative
from experiments.structured_fisher_shrinkage_01.mathematics import (
    TARGET_RANKS, _factor_inputs, _finite, _layout, estimate_shrinkage,
    global_absolute_ridge, structured_factor,
)


def nested_weights(eta_within: float, eta_cross: float) -> tuple[float, float, float]:
    """Validate the PSD ordering; callers estimate/clamp cross reliability once."""
    within, cross = float(eta_within), float(eta_cross)
    if not np.isfinite([within, cross]).all() or not 0 <= cross <= within <= 1:
        raise ValueError("hierarchical etas require 0 <= cross <= within <= 1")
    return cross, within - cross, 1.0 - within


@dataclass(frozen=True)
class HierarchicalEstimate:
    eta_within_raw: float
    eta_cross_raw: float
    eta_within: float
    eta_cross: float
    lambda_within: float
    lambda_cross: float
    within_mean_square_sum: float
    within_variance_sum: float
    within_sample_product_square_sum: float
    within_untruncated_lambda: float | None
    within_status: str
    cross_estimate_metadata: dict[str, Any]
    examples: int
    source_coordinates: int
    layer_count: int
    estimate_seconds: float
    cross_clipped_for_psd: bool
    role: str = "calibration"
    rank_independent: bool = True
    labels_used: bool = False
    outcomes_used: bool = False
    variance_definition: str = "sum_within_offdiag sum_i(z_ia*z_ib-F_ab)^2/[N*(N-1)]"
    nesting_rule: str = "eta_within=eta_within_raw; eta_cross=min(eta_cross_raw,eta_within)"

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def estimate_hierarchical(Z: Any, layer_slices: Mapping[str, slice], *,
                          role: str = "calibration") -> HierarchicalEstimate:
    """Estimate the two disjoint off-diagonal groups from source Fisher rows.

    Within-layer moments use only block-sized Grams and row energy moments.
    No per-example coordinate outer product or labels are constructed/read.
    Cross-layer estimation delegates to Issue 28 without changing its rule.
    """
    if role != "calibration":
        raise ValueError("hierarchical estimation reads calibration only")
    start = perf_counter()
    factor, ordered = _factor_inputs(Z, layer_slices, min_examples=2)
    n = len(factor)
    denominator, raw_products = 0.0, 0.0
    with threadpool_limits(limits=1):
        for _, sl in ordered:
            block = factor[:, sl]
            gram = block.T @ block
            # Explicitly zeroing just the Gram diagonal avoids subtracting two
            # nearly equal energies when within-layer coupling is absent.
            np.fill_diagonal(gram, 0.0)
            denominator += float(np.sum(gram * gram) / (n * n))
            squared = block * block
            energy = np.sum(squared, axis=1)
            all_products = float(np.dot(energy, energy))
            diagonal_products = float(np.sum(squared * squared))
            off_products = all_products - diagonal_products
            if off_products <= 64 * np.finfo(np.float64).eps * max(all_products, diagonal_products, 1e-300):
                # Rare cancellation fallback remains O(N*block_rank^2), not
                # an N-by-D-by-D collection of sample matrices.
                off_products = sum(float(2 * np.dot(squared[:, a], squared[:, b]))
                    for a in range(block.shape[1]) for b in range(a + 1, block.shape[1]))
            raw_products += off_products
        variance_numerator = raw_products - n * denominator
        roundoff = 128 * np.finfo(np.float64).eps * max(raw_products, n * denominator, 1e-300)
        if variance_numerator < -roundoff:
            raise ValueError("negative within-layer variance beyond floating-point roundoff")
        variance = max(variance_numerator, 0.0) / (n * (n - 1))
    if denominator == 0:
        ratio, shrinkage, within, status = None, 1.0, 0.0, "ZERO_WITHIN_OFFDIAGONAL_SIGNAL"
    else:
        ratio = float(variance / denominator)
        shrinkage = float(np.clip(ratio, 0.0, 1.0))
        within, status = 1.0 - shrinkage, "ANALYTIC_ESTIMATE"
    cross = estimate_shrinkage(factor, layer_slices, role=role)
    nested_cross = min(cross.eta, within)
    nested_weights(within, nested_cross)
    if not np.isfinite([denominator, raw_products, variance, within, shrinkage]).all():
        raise ValueError("non-finite hierarchical shrinkage moments")
    return HierarchicalEstimate(within, cross.eta, within, nested_cross, shrinkage,
        cross.lambda_hat, denominator, variance, raw_products, ratio, status,
        cross.metadata(), n, factor.shape[1], len(ordered), perf_counter() - start,
        cross.eta > within)


hierarchical_estimate = estimate_hierarchical


def hierarchical_gram(Z: Any, layer_slices: Mapping[str, slice],
                      eta_within: float, eta_cross: float) -> np.ndarray:
    """Explicit original-unit F_H for independent retained/pruned checks."""
    factor, ordered = _factor_inputs(Z, layer_slices)
    cross_weight, block_weight, diagonal_weight = nested_weights(eta_within, eta_cross)
    with threadpool_limits(limits=1):
        gram = factor.T @ factor / len(factor)
    result = cross_weight * gram
    for _, sl in ordered:
        result[sl, sl] += block_weight * gram[sl, sl]
    result[np.diag_indices_from(result)] += diagonal_weight * np.diag(gram)
    if not np.isfinite(result).all():
        raise ValueError("non-finite hierarchical Gram")
    return result


def hierarchical_factor(Z: Any, layer_slices: Mapping[str, slice],
                        eta_within: float, eta_cross: float) -> np.ndarray:
    """Return W with W.T@W/len(W)=cross*F+(within-cross)*B+(1-within)*D.

    A single row-count normalization scales all three roots alike, so both
    retained Hessian and retained/pruned term use exactly the same geometry.
    The within=1 endpoints reuse the established Issue-28 factor verbatim.
    """
    factor, ordered = _factor_inputs(Z, layer_slices)
    cross_weight, block_weight, diagonal_weight = nested_weights(eta_within, eta_cross)
    if float(eta_within) == 1.0:
        return structured_factor(factor, layer_slices, float(eta_cross))
    roots = []
    with threadpool_limits(limits=1):
        if block_weight > 0:
            roots = [np.linalg.qr(factor[:, sl], mode="reduced")[1] for _, sl in ordered]
    rows = (len(factor) if cross_weight > 0 else 0) + sum(len(r) for r in roots) + factor.shape[1]
    result = np.zeros((rows, factor.shape[1]), dtype=np.float64)
    offset = 0
    if cross_weight > 0:
        result[:len(factor)] = np.sqrt(cross_weight) * factor
        offset = len(factor)
    for (_, sl), block_root in zip(ordered, roots):
        result[offset:offset + len(block_root), sl] = np.sqrt(block_weight) * block_root
        offset += len(block_root)
    diagonal_root = np.sqrt(diagonal_weight * np.sum(factor * factor, axis=0))
    result[offset + np.arange(factor.shape[1]), np.arange(factor.shape[1])] = diagonal_root
    result *= np.sqrt(rows / len(factor))
    if not np.isfinite(result).all():
        raise ValueError("non-finite hierarchical factor")
    return result


def evaluate_hierarchical_objective(Z: Any, sigma: Any, layer_slices: Mapping[str, slice],
                                   keep_indices: Sequence[int], coefficients: Any, *,
                                   ridge: float, eta_within: float, eta_cross: float) -> dict[str, float]:
    """Evaluate candidate's own nested and common full source-state objectives."""
    factor, ordered = _factor_inputs(Z, layer_slices)
    source, candidate = _finite(sigma, "source coefficients"), _finite(coefficients, "candidate coefficients")
    cross_weight, block_weight, diagonal_weight = nested_weights(eta_within, eta_cross)
    raw_keep = np.asarray(keep_indices)
    if raw_keep.ndim != 1 or raw_keep.dtype.kind not in "iu" or not raw_keep.size:
        raise ValueError("keep indices must be a nonempty integer vector")
    keep = np.asarray(raw_keep, dtype=np.int64)
    if (source.shape != (factor.shape[1],) or candidate.shape != source.shape or
            np.any(source < 0) or np.any(candidate < 0) or np.any(keep < 0) or
            np.any(keep >= source.size) or len(np.unique(keep)) != len(keep)):
        raise ValueError("source/candidate coefficients or retained coordinates differ")
    tail = np.ones(source.size, dtype=bool)
    tail[keep] = False
    ridge = float(ridge)
    if np.any(candidate[tail] != 0) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("candidate tail must be zero and ridge positive finite")
    delta = candidate - source
    residual = factor @ delta
    full_Q = float(np.dot(residual, residual) / (2 * len(factor)))
    block_Q = 0.0
    for _, sl in ordered:
        residual = factor[:, sl] @ delta[sl]
        block_Q += float(np.dot(residual, residual) / (2 * len(factor)))
    diagonal_Q = float(np.dot(np.sum(factor * factor, axis=0), delta * delta) / (2 * len(factor)))
    own_Q = cross_weight * full_Q + block_weight * block_Q + diagonal_weight * diagonal_Q
    penalty = float(0.5 * ridge * np.dot(delta[keep], delta[keep]))
    values = {"structured_Q": float(own_Q), "structured_R": float(own_Q + penalty),
        "hierarchical_Q": float(own_Q), "hierarchical_R": float(own_Q + penalty),
        "common_Q": full_Q, "common_R": float(full_Q + penalty), "layerwise_Q": block_Q,
        "diagonal_Q": diagonal_Q, "ridge_penalty": penalty,
        "adjustment_norm": float(np.linalg.norm(delta[keep]))}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("non-finite hierarchical objective")
    return values


@dataclass(frozen=True)
class HierarchicalCondition:
    coefficients: np.ndarray
    delta: np.ndarray
    keep_indices: np.ndarray
    prune_indices: np.ndarray
    target_rank: int
    ridge: float
    eta_within: float
    eta_cross: float
    objective_R: float
    common_R: float
    Q_cal: float
    structured_Q: float
    adjustment_norm: float
    certificate: dict[str, Any]
    solver_metadata: dict[str, Any]
    factor_metadata: dict[str, Any]
    solve_seconds: float

    @property
    def keep(self) -> np.ndarray:
        return self.keep_indices

    @property
    def prune(self) -> np.ndarray:
        return self.prune_indices

    def diagnostics(self) -> dict[str, Any]:
        return {"target_rank": self.target_rank, "ridge": self.ridge,
            "eta_within": self.eta_within, "eta_cross": self.eta_cross,
            "optimization_R": self.objective_R, "objective_R": self.objective_R,
            "common_R": self.common_R, "Q_cal": self.Q_cal, "structured_Q": self.structured_Q,
            "adjustment_norm": self.adjustment_norm, "kept_coefficients": int(self.keep_indices.size),
            "source_coordinates": int(self.coefficients.size), "certificate": self.certificate,
            "solver_metadata": self.solver_metadata, "factor_metadata": self.factor_metadata,
            "solve_seconds": self.solve_seconds, "coefficient_constraint": "nonnegative",
            "solver_reused": "Issue-27 unchanged exact NNLS", "both_quadratic_and_cross_term_shrunk": True}


def _condition_from_factor(Z: Any, sigma: Any, layer_slices: Mapping[str, slice], target_rank: int, *,
                           ridge: float, eta_within: float, eta_cross: float, W: np.ndarray) -> HierarchicalCondition:
    start = perf_counter()
    factor, ordered = _factor_inputs(Z, layer_slices)
    source = _finite(sigma, "source coefficients")
    nested_weights(eta_within, eta_cross)
    if source.shape != (factor.shape[1],) or np.any(source < 0):
        raise ValueError("source coefficient shape/sign mismatch")
    if type(target_rank) is not int or target_rank not in TARGET_RANKS:
        raise ValueError("frozen target ranks are 8,4,2")
    _layout(layer_slices, source.size, source_rank=32)
    keep = np.concatenate([np.arange(sl.start, sl.start + target_rank, dtype=np.int64) for _, sl in ordered])
    solved = solve_nonnegative(W, source, keep, ridge=ridge)
    candidate = np.zeros_like(source)
    candidate[keep] = solved.c_keep
    tail = np.ones(source.size, dtype=bool)
    tail[keep] = False
    values = evaluate_hierarchical_objective(factor, source, layer_slices, keep, candidate,
        ridge=ridge, eta_within=eta_within, eta_cross=eta_cross)
    tolerance = 256 * np.finfo(np.float64).eps * max(values["structured_R"], solved.R, 1e-12)
    if abs(values["structured_R"] - solved.R) > tolerance:
        raise ValueError("factor objective disagrees with nested original-unit objective")
    return HierarchicalCondition(candidate, candidate - source, keep, np.flatnonzero(tail),
        target_rank, float(ridge), float(eta_within), float(eta_cross), values["structured_R"],
        values["common_R"], values["common_Q"], values["structured_Q"], values["adjustment_norm"],
        dict(solved.certificate, objective_scope="hierarchical F/B/D empirical objective"),
        solved.solver_metadata, {"original_examples": len(factor), "factor_rows": len(W),
            "source_coordinates": source.size,
            "construction": "sqrt(cross)*Z + sqrt(within-cross)*block QR roots + sqrt(1-within)*diagonal root",
            "normalization_rescaling": float(np.sqrt(len(W) / len(factor))),
            "weights": list(nested_weights(eta_within, eta_cross)), "retained_pruned_term_shrunk": True},
        perf_counter() - start)


def build_condition(Z: Any, sigma: Any, layer_slices: Mapping[str, slice], target_rank: int, *,
                    ridge: float, eta_within: float, eta_cross: float) -> HierarchicalCondition:
    """Solve one genuine-rank coefficient condition at the supplied frozen ridge."""
    start = perf_counter()
    W = hierarchical_factor(Z, layer_slices, eta_within, eta_cross)
    result = _condition_from_factor(Z, sigma, layer_slices, target_rank, ridge=ridge,
        eta_within=eta_within, eta_cross=eta_cross, W=W)
    return HierarchicalCondition(**{**result.__dict__, "solve_seconds": perf_counter() - start})


build_hierarchical_condition = build_condition
