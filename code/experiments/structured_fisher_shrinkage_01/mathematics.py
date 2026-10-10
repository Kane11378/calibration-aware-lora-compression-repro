"""Source-only analytic Fisher shrinkage and unchanged Issue-27 NNLS solves.

Only frozen source spectral coefficients, the calibration GGN factor, fixed
layer coordinates, and supplied ridge scalars enter deployed compensation.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.constrained_compensation_control_01.solver import solve_nonnegative

TARGET_RANKS = (8, 4, 2)
CROSSFIT_PERMUTATION_SEED = 20261005
RELATIVE_RIDGE = 1e-3
RIDGE_SCALE_FLOOR = 1e-12


def _finite(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite {name}")
    return result


def _layout(layer_slices: Mapping[str, slice], dimension: int, *, source_rank: int | None = None):
    start = 0
    ordered = []
    for name in sorted(layer_slices):
        sl = layer_slices[name]
        if (not isinstance(sl, slice) or sl.start != start or not isinstance(sl.stop, int) or
                sl.stop <= start or sl.step not in (None, 1) or
                (source_rank is not None and sl.stop - start != source_rank)):
            raise ValueError("layer map must cover contiguous sorted source coordinates")
        ordered.append((name, sl))
        start = sl.stop
    if start != dimension or not ordered:
        raise ValueError("layer map must cover all source coordinates")
    return ordered


def _factor_inputs(Z: Any, layer_slices: Mapping[str, slice], *, min_examples: int = 1):
    factor = _finite(Z, "calibration factor")
    if factor.ndim != 2 or factor.shape[0] < min_examples or factor.shape[1] < 1:
        raise ValueError(f"calibration factor must be nonempty [N,D] with N>={min_examples}")
    ordered = _layout(layer_slices, factor.shape[1])
    return factor, ordered


def _eta(value: float) -> float:
    eta = float(value)
    if not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite and in [0,1]")
    return eta


@dataclass(frozen=True)
class ShrinkageEstimate:
    eta: float
    lambda_hat: float
    offblock_mean_square_sum: float
    offblock_variance_sum: float
    offblock_sample_product_square_sum: float
    untruncated_lambda: float | None
    examples: int
    source_coordinates: int
    layer_count: int
    status: str
    estimate_seconds: float
    role: str = "calibration"
    rank_independent: bool = True
    moment_definition: str = "uncentered projected Fisher second moment"
    variance_definition: str = "sum_offblock sum_i(z_ia*z_ib-F_ab)^2/[N*(N-1)]"
    implementation: str = "sample Gram plus layer Gram and row block energy moments"

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def estimate_shrinkage(Z: Any, layer_slices: Mapping[str, slice], *, role: str = "calibration") -> ShrinkageEstimate:
    """Compute the rank-independent analytic scalar without labels or tuning.

    No per-example D-by-D outer product is formed. The global mean-square sum
    uses an N-by-N sample Gram; all layer Grams remain block-sized.
    """
    if role != "calibration":
        raise ValueError("analytic shrinkage estimation reads calibration only")
    start = perf_counter()
    factor, ordered = _factor_inputs(Z, layer_slices, min_examples=2)
    n = len(factor)
    with threadpool_limits(limits=1):
        sample_gram = factor @ factor.T
        all_mean_square = float(np.sum(sample_gram * sample_gram) / (n * n))
        within_mean_square = 0.0
        row_block_energy = np.empty((n, len(ordered)), dtype=np.float64)
        for index, (_, sl) in enumerate(ordered):
            block = factor[:, sl]
            gram = block.T @ block
            within_mean_square += float(np.sum(gram * gram) / (n * n))
            row_block_energy[:, index] = np.sum(block * block, axis=1)
        denominator = all_mean_square - within_mean_square
        cancellation_scale = 64 * np.finfo(np.float64).eps * max(all_mean_square, within_mean_square, 1e-300)
        if denominator <= cancellation_scale:
            # Resolve rare cancellation by summing nonnegative cross-block
            # Gram energies; still no per-example coordinate outer products.
            denominator = 0.0
            for i, (_, first) in enumerate(ordered):
                for _, second in ordered[i + 1:]:
                    cross = factor[:, first].T @ factor[:, second]
                    denominator += float(2 * np.sum(cross * cross) / (n * n))
        row_energy = np.sum(row_block_energy, axis=1)
        raw_offblock_products = float(np.sum(row_energy * row_energy) -
                                      np.sum(row_block_energy * row_block_energy))
        if raw_offblock_products <= 64 * np.finfo(np.float64).eps * max(float(np.sum(row_energy * row_energy)), 1e-300):
            raw_offblock_products = sum(float(2 * np.dot(row_block_energy[:, i], row_block_energy[:, j]))
                                        for i in range(len(ordered)) for j in range(i + 1, len(ordered)))
        variance_numerator = raw_offblock_products - n * denominator
        variance_roundoff = 128 * np.finfo(np.float64).eps * max(raw_offblock_products, n * denominator, 1e-300)
        if variance_numerator < -variance_roundoff:
            raise ValueError("negative offblock variance beyond floating-point roundoff")
        variance_sum = max(variance_numerator, 0.0) / (n * (n - 1))
    if denominator == 0:
        # Zero empirical off-block energy supplies no support for retaining it.
        lambda_hat, eta, ratio, status = 1.0, 0.0, None, "ZERO_OFFBLOCK_SIGNAL"
    else:
        ratio = float(variance_sum / denominator)
        lambda_hat = float(np.clip(ratio, 0.0, 1.0))
        eta, status = 1.0 - lambda_hat, "ANALYTIC_ESTIMATE"
    if not np.isfinite([denominator, variance_sum, raw_offblock_products, eta, lambda_hat]).all():
        raise ValueError("non-finite analytic shrinkage moments")
    return ShrinkageEstimate(eta, lambda_hat, float(denominator), float(variance_sum),
                             float(raw_offblock_products), ratio, n, factor.shape[1], len(ordered),
                             status, perf_counter() - start)


def structured_factor(Z: Any, layer_slices: Mapping[str, slice], eta: float) -> np.ndarray:
    """Construct W with W.T@W/len(W)=eta*Fhat+(1-eta)*blockdiag(Fhat).

    QR roots represent each unchanged source-layer Gram. The single factor-row
    count rescaling compensates the Issue-27 solver's original normalization;
    it applies equally to both retained-retained and retained-pruned blocks.
    """
    factor, ordered = _factor_inputs(Z, layer_slices)
    weight = _eta(eta)
    if weight == 1:
        return np.array(factor, dtype=np.float64, copy=True)
    roots = []
    with threadpool_limits(limits=1):
        for _, sl in ordered:
            _, root = np.linalg.qr(factor[:, sl], mode="reduced")
            roots.append(root)
    total_rows = len(factor) + sum(len(root) for root in roots)
    W = np.zeros((total_rows, factor.shape[1]), dtype=np.float64)
    W[:len(factor)] = np.sqrt(weight) * factor
    offset = len(factor)
    for (_, sl), root in zip(ordered, roots):
        W[offset:offset + len(root), sl] = np.sqrt(1.0 - weight) * root
        offset += len(root)
    W *= np.sqrt(total_rows / len(factor))
    if not np.isfinite(W).all():
        raise ValueError("non-finite structured factor")
    return W


def global_absolute_ridge(Z: Any, keep_indices: Sequence[int]) -> float:
    """The unchanged Issue-25/27 relative ridge rule, used on train folds only."""
    factor = _finite(Z, "factor")
    raw = np.asarray(keep_indices)
    if factor.ndim != 2 or len(factor) < 1 or raw.ndim != 1 or raw.dtype.kind not in "iu" or not raw.size:
        raise ValueError("invalid ridge factor/keep indices")
    keep = np.asarray(raw, dtype=np.int64)
    if np.any(keep < 0) or np.any(keep >= factor.shape[1]) or len(np.unique(keep)) != len(keep):
        raise ValueError("invalid ridge keep indices")
    mean_diagonal = float(np.mean(np.sum(np.square(factor[:, keep]), axis=0) / len(factor)))
    return float(RELATIVE_RIDGE * max(mean_diagonal, RIDGE_SCALE_FLOOR))


def evaluate_condition_objective(Z: Any, sigma: Any, layer_slices: Mapping[str, slice],
                                 keep_indices: Sequence[int], coefficients: Any,
                                 *, ridge: float, eta: float) -> dict[str, float]:
    """Evaluate both the shrunken own objective and common unshrunken objective."""
    factor, ordered = _factor_inputs(Z, layer_slices)
    source = _finite(sigma, "source coefficients")
    candidate = _finite(coefficients, "candidate coefficients")
    weight = _eta(eta)
    d = float(ridge)
    keep_raw = np.asarray(keep_indices)
    if source.shape != (factor.shape[1],) or candidate.shape != source.shape or np.any(source < 0) or np.any(candidate < 0):
        raise ValueError("invalid full source/candidate coefficients")
    if keep_raw.ndim != 1 or keep_raw.dtype.kind not in "iu" or not keep_raw.size:
        raise ValueError("keep indices must be a nonempty integer vector")
    keep = np.asarray(keep_raw, dtype=np.int64)
    if np.any(keep < 0) or np.any(keep >= source.size) or len(np.unique(keep)) != len(keep):
        raise ValueError("invalid keep indices")
    mask = np.ones(source.size, dtype=bool)
    mask[keep] = False
    if np.any(candidate[mask] != 0) or not np.isfinite(d) or d <= 0:
        raise ValueError("candidate tail must be zero and ridge positive finite")
    delta = candidate - source
    residual = factor @ delta
    global_Q = float(np.dot(residual, residual) / (2 * len(factor)))
    local_Q = 0.0
    for _, sl in ordered:
        residual = factor[:, sl] @ delta[sl]
        local_Q += float(np.dot(residual, residual) / (2 * len(factor)))
    penalty = float(0.5 * d * np.dot(delta[keep], delta[keep]))
    structured_Q = weight * global_Q + (1 - weight) * local_Q
    values = {"structured_Q": float(structured_Q), "structured_R": float(structured_Q + penalty),
              "common_Q": global_Q, "common_R": float(global_Q + penalty),
              "layerwise_Q": local_Q, "ridge_penalty": penalty,
              "adjustment_norm": float(np.linalg.norm(delta[keep]))}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("non-finite condition objective")
    return values


@dataclass(frozen=True)
class ShrinkageCondition:
    coefficients: np.ndarray
    delta: np.ndarray
    keep_indices: np.ndarray
    prune_indices: np.ndarray
    target_rank: int
    ridge: float
    eta: float
    objective_R: float
    common_R: float
    Q_cal: float
    structured_Q: float
    adjustment_norm: float
    certificate: dict[str, Any]
    solver_metadata: dict[str, Any]
    factor_metadata: dict[str, Any]
    solve_seconds: float

    def diagnostics(self) -> dict[str, Any]:
        return {"target_rank": self.target_rank, "ridge": self.ridge, "eta": self.eta,
                "lambda_hat": 1.0 - self.eta, "optimization_R": self.objective_R,
                "objective_R": self.objective_R, "common_R": self.common_R, "Q_cal": self.Q_cal,
                "structured_Q": self.structured_Q, "adjustment_norm": self.adjustment_norm,
                "kept_coefficients": int(self.keep_indices.size), "source_coordinates": int(self.coefficients.size),
                "certificate": self.certificate, "solver_metadata": self.solver_metadata,
                "factor_metadata": self.factor_metadata, "solve_seconds": self.solve_seconds,
                "coefficient_constraint": "nonnegative", "solver_reused": "Issue-27 unchanged exact NNLS",
                "both_quadratic_and_cross_term_shrunk": True}


def _condition_from_factor(Z, sigma, layer_slices, target_rank, *, ridge, eta, W):
    start = perf_counter()
    factor, ordered = _factor_inputs(Z, layer_slices)
    source = _finite(sigma, "source coefficients")
    if source.shape != (factor.shape[1],) or np.any(source < 0):
        raise ValueError("source coefficient shape/sign mismatch")
    if target_rank not in TARGET_RANKS:
        raise ValueError("frozen target ranks are 8,4,2")
    _layout(layer_slices, source.size, source_rank=32)
    keep = np.concatenate([np.arange(sl.start, sl.start + target_rank, dtype=np.int64) for _, sl in ordered])
    solved = solve_nonnegative(W, source, keep, ridge=ridge)
    coefficients = np.zeros_like(source)
    coefficients[keep] = solved.c_keep
    mask = np.ones(source.size, dtype=bool)
    mask[keep] = False
    values = evaluate_condition_objective(factor, source, layer_slices, keep, coefficients, ridge=ridge, eta=eta)
    # This algebra check certifies both retained Hessian and retained-pruned RHS
    # under the same geometry; it never uses task or tuning outcomes.
    tolerance = 256 * np.finfo(np.float64).eps * max(values["structured_R"], solved.R, 1e-12)
    if abs(values["structured_R"] - solved.R) > tolerance:
        raise ValueError("structured-factor objective disagrees with frozen shrunken objective")
    return ShrinkageCondition(coefficients, coefficients - source, keep, np.flatnonzero(mask),
        target_rank, float(ridge), float(eta), values["structured_R"], values["common_R"], values["common_Q"],
        values["structured_Q"], values["adjustment_norm"],
        dict(solved.certificate, objective_scope="structured F_eta empirical objective"), solved.solver_metadata,
        {"original_examples": len(factor), "factor_rows": len(W), "source_coordinates": source.size,
         "construction": "sqrt(eta)*Z plus sqrt(1-eta)*block QR roots; common row-count rescaling",
         "normalization_rescaling": float(np.sqrt(len(W) / len(factor))),
         "unchanged_source_layer_blocks": True, "retained_pruned_term_shrunk": True},
        perf_counter() - start)


def build_condition(Z: Any, sigma: Any, layer_slices: Mapping[str, slice], target_rank: int,
                    *, ridge: float, eta: float) -> ShrinkageCondition:
    """Materialize one coefficient vector; supplied deployment ridge stays fixed."""
    start = perf_counter()
    weight = _eta(eta)
    W = structured_factor(Z, layer_slices, weight)
    result = _condition_from_factor(Z, sigma, layer_slices, target_rank, ridge=ridge, eta=weight, W=W)
    # Include factor construction in the reported solver pipeline cost.
    return ShrinkageCondition(**{**result.__dict__, "solve_seconds": perf_counter() - start})


def crossfit_diagnostic(Z: Any, sigma: Any, layer_slices: Mapping[str, slice],
                       *, source_seed: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fixed calibration-only 4-fold diagnostic; never chooses deployment eta."""
    factor, ordered = _factor_inputs(Z, layer_slices)
    source = _finite(sigma, "source coefficients")
    if len(factor) != 512 or source.shape != (factor.shape[1],):
        raise ValueError("crossfit requires frozen 512 calibration rows and matching source coefficients")
    _layout(layer_slices, source.size, source_rank=32)
    permutation = np.random.default_rng(CROSSFIT_PERMUTATION_SEED).permutation(512)
    detailed = []
    for fold in range(4):
        held_positions = permutation[fold * 128:(fold + 1) * 128]
        train_positions = np.setdiff1d(np.arange(512), held_positions)
        train, held = factor[train_positions], factor[held_positions]
        estimate = estimate_shrinkage(train, layer_slices)
        factors = {"C1": structured_factor(train, layer_slices, 0.0),
                   "S1": structured_factor(train, layer_slices, estimate.eta),
                   "C2": train}
        for rank in TARGET_RANKS:
            keep = np.concatenate([np.arange(sl.start, sl.start + rank, dtype=np.int64) for _, sl in ordered])
            ridge = global_absolute_ridge(train, keep)
            for method, weight in (("C1", 0.0), ("S1", estimate.eta), ("C2", 1.0)):
                condition = _condition_from_factor(train, source, layer_slices, rank, ridge=ridge,
                                                   eta=weight, W=factors[method])
                held_residual = held @ condition.delta
                Q_held = float(np.dot(held_residual, held_residual) / (2 * len(held)))
                detailed.append({"seed": source_seed, "target_rank": rank, "fold": fold, "method": method,
                    "eta": weight, "analytic_train_eta": estimate.eta, "analytic_train_lambda": estimate.lambda_hat,
                    "ridge": ridge, "ridge_role": "recomputed from training fold only; original 1e-3/floor1e-12 rule",
                    "train_examples": 384, "held_examples": 128, "Q_train": condition.Q_cal,
                    "Q_train_structured": condition.structured_Q, "R_train_structured": condition.objective_R,
                    "Q_held": Q_held, "adjustment_norm": condition.adjustment_norm,
                    "scaled_kkt_residual": condition.certificate["scaled_kkt_residual"],
                    "primal_dual_gap": condition.certificate["primal_dual_gap"],
                    "gap_tolerance": condition.certificate["gap_tolerance"], "certificate_passed": condition.certificate["passed"],
                    "permutation_seed": CROSSFIT_PERMUTATION_SEED,
                    "train_positions": train_positions.tolist(), "held_positions": held_positions.tolist(),
                    "train_factor_sha256": hashlib.sha256(np.asarray(train, dtype="<f8").tobytes()).hexdigest(),
                    "diagnostic_only": True, "deployment_eta_selected": False, "solve_seconds": condition.solve_seconds,
                    "shrinkage_estimate_metadata": estimate.metadata(), "certificate": condition.certificate})
    pooled = []
    for rank in TARGET_RANKS:
        for method in ("C1", "S1", "C2"):
            records = [row for row in detailed if row["target_rank"] == rank and row["method"] == method]
            pooled.append({"seed": source_seed, "target_rank": rank, "method": method, "folds": 4,
                           "held_examples": sum(row["held_examples"] for row in records),
                           "pooled_Q_held": float(sum(row["Q_held"] * row["held_examples"] for row in records) / 512),
                           "train_eta_mean": float(np.mean([row["analytic_train_eta"] for row in records])),
                           "train_eta_min": float(min(row["analytic_train_eta"] for row in records)),
                           "train_eta_max": float(max(row["analytic_train_eta"] for row in records)),
                           "permutation_seed": CROSSFIT_PERMUTATION_SEED,
                           "diagnostic_only": True, "deployment_eta_selected": False})
    return detailed, pooled
