"""
Experimentation Platform: CUPED Variance Reduction.

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the
variance of the treatment effect estimator by regressing out variance
explained by a pre-experiment covariate (e.g. the user's metric value
from the week before the experiment started).

Lower variance → tighter confidence intervals → smaller required sample
size to detect the same effect. In practice, CUPED typically reduces
required sample size by 20–50% depending on the covariate's correlation
with the outcome metric.

Key reference:
    Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013).
    Improving the sensitivity of online controlled experiments
    by utilizing pre-experiment data. WSDM 2013.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.readout import ReadoutStrategy, ReadoutResult, TTestReadout


# ---------------------------------------------------------------------------
# 1. CUPED result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CUPEDResult:
    """Immutable container comparing raw vs. CUPED-adjusted readout outputs.

    Stores both the unadjusted and CUPED-adjusted ReadoutResults so the
    caller can directly compare variance reduction and its downstream
    effect on confidence interval width and statistical power.

    Attributes:
        raw_result: ReadoutResult computed on the original, unadjusted
            metric observations.
        adjusted_result: ReadoutResult computed on the CUPED-adjusted
            metric observations (residuals after covariate regression).
        theta: The regression coefficient used for covariate adjustment.
            Estimated from pooled control + treatment data to avoid
            using the treatment assignment in its estimation.
        covariate_correlation: Pearson correlation between the covariate
            and the outcome metric, pooled across both groups. Higher
            correlation → greater variance reduction.
        variance_reduction_pct: Percentage reduction in metric variance
            achieved by CUPED adjustment relative to the raw variance.
            Computed as (1 - var(adjusted) / var(raw)) * 100.
    """
    raw_result: ReadoutResult
    adjusted_result: ReadoutResult
    theta: float
    covariate_correlation: float
    variance_reduction_pct: float

    def summary(self) -> str:
        """Return a human-readable comparison of raw vs. adjusted readouts.

        Returns:
            Multi-line string covering covariate correlation, theta,
            variance reduction, and a side-by-side comparison of
            p-values and CI widths before and after adjustment.
        """
        # TODO: Format and return a report string covering:
        #   - covariate_correlation and theta
        #   - variance_reduction_pct
        #   - raw p_value and CI width (raw_result.ci_upper - raw_result.ci_lower)
        #   - adjusted p_value and CI width (adjusted_result.ci_upper - adjusted_result.ci_lower)
        #   - plain-English note on whether adjustment improved significance:
        #       e.g. "CUPED tightened the CI by X% and reduced p-value from A to B."
        ...


# ---------------------------------------------------------------------------
# 2. CUPED adjuster
# ---------------------------------------------------------------------------

class CUPEDAdjuster:
    """Applies CUPED covariate adjustment to reduce metric variance.

    CUPED works by subtracting from each user's observed metric the
    portion of variance predictable from a pre-experiment covariate X
    (e.g. the same metric measured in a pre-experiment window):

        Y_adjusted = Y - theta * (X - E[X])

    where theta minimises Var(Y_adjusted), which yields:

        theta = Cov(Y, X) / Var(X)

    Because E[X] is the same constant subtracted from both groups,
    the expected value of Y_adjusted equals the expected value of Y --
    adjustment is variance-reducing without introducing bias.

    theta is estimated from pooled (control + treatment) data to avoid
    using treatment assignment in its estimation, which would introduce
    bias if the experiment has any true effect.
    """

    def __init__(self, readout: ReadoutStrategy = None):
        """Initialise the CUPED adjuster with an optional readout strategy.

        Args:
            readout: ReadoutStrategy instance used to compute both the
                raw and adjusted ReadoutResults. Defaults to
                TTestReadout(alpha=0.05) if not provided.
        """
        if readout is None:
            self._readout = TTestReadout()
        else:
            self._readout = readout

    def _estimate_theta(
        self,
        outcome: np.ndarray,
        covariate: np.ndarray,
    ) -> float:
        """Estimate the optimal CUPED regression coefficient theta.

        Theta is the OLS coefficient from regressing the outcome on the
        covariate, estimated on pooled data (both groups combined) to
        ensure unbiasedness under the null and under alternatives.

        Args:
            outcome: Pooled outcome metric observations (control + treatment
                concatenated).
            covariate: Pooled pre-experiment covariate values in the same
                order as outcome.

        Returns:
            Scalar theta = Cov(outcome, covariate) / Var(covariate).

        Raises:
            ValueError: If covariate has zero variance (constant covariate
                provides no information and makes theta undefined).
        """
        covariate_var = np.var(covariate, ddof=1)
        if covariate_var == 0:
            raise ValueError("Covariate has zero variance; CUPED adjustment is undefined.")
        
        # covariate_mean = np.mean(covariate)
        outcome_cov = np.cov(outcome, covariate, ddof=1)[0, 1] #note: np.cov returns a 2x2 matrix; [0,1] is the cross-covariance
        theta = outcome_cov / covariate_var

        return theta

    def _adjust(
        self,
        metric: np.ndarray,
        covariate: np.ndarray,
        theta: float,
        covariate_mean: float,
    ) -> np.ndarray:
        """Apply CUPED adjustment to a single group's metric observations.

        Args:
            metric: Raw outcome metric observations for one group.
            covariate: Pre-experiment covariate values for the same group,
                in the same user order as metric.
            theta: Regression coefficient estimated from pooled data.
            covariate_mean: Pooled covariate mean E[X], subtracted to
                preserve the expected value of the adjusted metric.

        Returns:
            1-D array of CUPED-adjusted metric values, same length as metric.
        """
        return metric - theta * (covariate - covariate_mean)

    def compute(
        self,
        control_metric: np.ndarray,
        treatment_metric: np.ndarray,
        control_covariate: np.ndarray,
        treatment_covariate: np.ndarray,
    ) -> CUPEDResult:
        """Run CUPED adjustment and return raw vs. adjusted readout comparison.

        Estimates theta from pooled data, adjusts both groups, then runs
        the configured readout strategy on both the original and adjusted
        observations to quantify the variance reduction achieved.

        Args:
            control_metric: Raw outcome metric for the control group.
            treatment_metric: Raw outcome metric for the treatment group.
            control_covariate: Pre-experiment covariate for the control group,
                aligned user-by-user with control_metric.
            treatment_covariate: Pre-experiment covariate for the treatment group,
                aligned user-by-user with treatment_metric.

        Returns:
            CUPEDResult containing raw result, adjusted result, theta,
            covariate correlation, and variance reduction percentage.

        Raises:
            ValueError: If any input array is empty, contains non-finite
                values, or if covariate arrays are misaligned with metric
                arrays (different lengths within a group).
        """
        if len(control_metric) != len(control_covariate): 
            raise ValueError ("Control metric and covariate must be the same length.")
        if len(treatment_metric) != len(treatment_covariate): 
            raise ValueError ("Treatment metric and covariate must be the same length.")
        if np.any(~np.isfinite(control_metric)) or np.any(~np.isfinite(treatment_metric)):
            raise ValueError("Input arrays must not contain NaN or Inf values.")
        
        raw_result = self._readout.compute(control_metric, treatment_metric)

        pooled_outcome = np.concatenate([control_metric, treatment_metric])
        pooled_covariate = np.concatenate([control_covariate, treatment_covariate])

        theta = self._estimate_theta(pooled_outcome, pooled_covariate)
        covariate_mean = np.mean(pooled_covariate)
        control_adjusted = self._adjust(control_metric, control_covariate, theta, covariate_mean)
        treatment_adjusted = self._adjust(treatment_metric, treatment_covariate, theta, covariate_mean)
        adjusted_result = self._readout.compute(control_adjusted, treatment_adjusted)

        raw_var = np.var(np.concatenate([control_metric, treatment_metric]), ddof=1)
        adjusted_var = np.var(np.concatenate([control_adjusted, treatment_adjusted]), ddof=1)
        variance_reduction_pct = (1 - adjusted_var / raw_var) * 100

        covariate_correlation = np.corrcoef(pooled_outcome, pooled_covariate)[0, 1]
        
        return CUPEDResult(
                 raw_result=raw_result,
                 adjusted_result=adjusted_result,
                 theta=theta,
                 covariate_correlation=covariate_correlation,
                 variance_reduction_pct=variance_reduction_pct,
             )
