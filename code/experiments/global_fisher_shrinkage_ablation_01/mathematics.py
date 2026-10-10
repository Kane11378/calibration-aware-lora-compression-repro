"""Issue43 G1: one pooled analytic off-diagonal calibration reliability.

The two inherited moment groups are disjoint and are added without retuning.
Factor construction and the constrained solve use the locked H1 utilities at
within=cross=g, whose empirical geometry is exactly g F + (1-g) D.
No labels, held-out endpoints, paths or candidate outcomes enter this API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from experiments.hierarchical_fisher_shrinkage_01.shrinkage import (
    HierarchicalCondition, HierarchicalEstimate, build_condition as _locked_condition,
    estimate_hierarchical, hierarchical_factor, hierarchical_gram,
)
from experiments.structured_fisher_shrinkage_01.mathematics import _eta


@dataclass(frozen=True)
class GlobalEstimate:
    g: float
    lambda_global: float
    pooled_mean_square_sum: float
    pooled_variance_sum: float
    pooled_sample_product_square_sum: float
    untruncated_lambda: float | None
    status: str
    within_mean_square_sum: float
    within_variance_sum: float
    offblock_mean_square_sum: float
    offblock_variance_sum: float
    inherited_hierarchical_metadata: dict[str, Any]
    examples: int
    source_coordinates: int
    layer_count: int
    role: str = "calibration"
    rank_independent: bool = True
    labels_used: bool = False
    outcomes_used: bool = False
    variance_definition: str = "sum_a!=b sum_i(z_ia*z_ib-F_ab)^2/[N*(N-1)]"
    pooling_rule: str = "S_G=S_W+S_C; V_G=V_W+V_C; g=1-clip(V_G/S_G,0,1); g=0 if S_G=0"

    @property
    def pooled_signal(self) -> float:
        return self.pooled_mean_square_sum

    @property
    def pooled_variance(self) -> float:
        return self.pooled_variance_sum

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def pooled_from_hierarchical(estimate: HierarchicalEstimate) -> GlobalEstimate:
    """Pool the exact already-computed H1 within and S1 off-block moments."""
    if not isinstance(estimate, HierarchicalEstimate) or estimate.role != "calibration":
        raise ValueError("G1 pooling requires the inherited calibration estimate")
    if estimate.labels_used or estimate.outcomes_used:
        raise ValueError("G1 moment pooling forbids labels and outcomes")
    cross = estimate.cross_estimate_metadata
    within_signal = estimate.within_mean_square_sum
    within_variance = estimate.within_variance_sum
    cross_signal = cross["offblock_mean_square_sum"]
    cross_variance = cross["offblock_variance_sum"]
    signal = within_signal + cross_signal
    variance = within_variance + cross_variance
    raw_products = estimate.within_sample_product_square_sum + cross["offblock_sample_product_square_sum"]
    values = [within_signal, within_variance, cross_signal, cross_variance, signal, variance, raw_products]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("non-finite or negative inherited G1 moments")
    if signal == 0:
        ratio, shrinkage, g, status = None, 1.0, 0.0, "ZERO_OFFDIAGONAL_SIGNAL"
    else:
        ratio = float(variance / signal)
        shrinkage = float(np.clip(ratio, 0.0, 1.0))
        g, status = 1.0 - shrinkage, "ANALYTIC_ESTIMATE"
    _eta(g)
    return GlobalEstimate(g, shrinkage, signal, variance, raw_products, ratio, status,
                          within_signal, within_variance, cross_signal, cross_variance,
                          estimate.metadata(), estimate.examples, estimate.source_coordinates,
                          estimate.layer_count)


def estimate_global(Z: Any, layer_slices: Mapping[str, slice], *, role: str = "calibration") -> GlobalEstimate:
    """Estimate one rank-independent g from frozen source calibration rows."""
    if role != "calibration":
        raise ValueError("G1 estimation reads calibration only")
    return pooled_from_hierarchical(estimate_hierarchical(Z, layer_slices, role=role))


def global_gram(Z: Any, layer_slices: Mapping[str, slice], g: float) -> np.ndarray:
    """Original-units g F + (1-g) D through unchanged locked geometry."""
    weight = _eta(g)
    return hierarchical_gram(Z, layer_slices, weight, weight)


def global_factor(Z: Any, layer_slices: Mapping[str, slice], g: float) -> np.ndarray:
    """Locked factor W, normalized so W.T W/len(W) = g F + (1-g) D."""
    weight = _eta(g)
    return hierarchical_factor(Z, layer_slices, weight, weight)


@dataclass(frozen=True)
class GlobalCondition(HierarchicalCondition):
    """The unchanged H1 condition object at equal within/cross weights."""
    @property
    def g(self) -> float:
        return self.eta_cross

    def diagnostics(self) -> dict[str, Any]:
        return {**super().diagnostics(), "method": "G1", "g": self.g,
                "geometry": "g*F+(1-g)*D", "within_cross_weights_identical": True,
                "factor_and_solver_reused": "locked H1 with eta_within=eta_cross=g"}


def build_condition(Z: Any, sigma: Any, layer_slices: Mapping[str, slice], target_rank: int,
                    *, ridge: float, g: float) -> GlobalCondition:
    """Use the supplied archived ridge, fixed support and original solver."""
    weight = _eta(g)
    result = _locked_condition(Z, sigma, layer_slices, target_rank, ridge=ridge,
                               eta_within=weight, eta_cross=weight)
    return GlobalCondition(**result.__dict__)


estimate_global_shrinkage = estimate_global
build_global_condition = build_condition
