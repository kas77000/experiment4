"""Tier 4 knobs --- VWAP-native thresholds."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier4Config:
    # --- 1) benchmark de-biasing -----------------------------------------
    # Divide reported slippage by (1 - participation) to measure against the
    # rest of the market rather than against a benchmark you partly constitute.
    # Exact algebra, not a model. See metric.py for the derivation.
    debias_benchmark: bool = True

    # Cap on the participation used in the correction. At f = 0.9 the factor is
    # 10x, and a participation figure that high is usually unreliable anyway.
    # Capped rows are marked, not silently adjusted.
    max_dilution_participation: float = 0.50   # -> correction capped at x2

    # --- 2) the tracking-error scale --------------------------------------
    # sigma_track = sqrt( (k_spread*spread)^2 + (k_track*vol*sqrt(T/S))^2 )
    #
    # Deliberately weighted differently from Tier 3. For a VWAP order the error
    # is schedule deviation interacting with the intraday price path, so the
    # volatility-over-horizon term should dominate; the spread term only covers
    # the per-child-order cost. Tier 3 uses 0.5 / 0.20.
    k_spread: float = 0.25
    k_track: float = 0.35

    # --- 3) the band ------------------------------------------------------
    tau_lo: float = 0.01
    tau_med: float = 0.50
    tau_hi: float = 0.995

    algo_effect: str = "absorb"

    # Size is KEPT but demoted -- no sqrt(%ADV) x POV interaction, and no POV
    # term at all. For a VWAP algo participation is an output of the volume
    # curve, not an urgency decision, so modelling it as one imports a POV/IS
    # framing that does not apply.
    #
    # Testable prediction: after de-biasing, the sqrt_adv coefficient should
    # collapse toward zero. If it does not, there is residual impact on the
    # non-self portion of the benchmark, which is worth knowing either way.
    # run.py prints the Tier 3 and Tier 4 coefficients side by side so you can
    # check rather than assume.
    include_size: bool = True

    n_folds: int = 5
    fit_trim_quantile: float = 0.0
    min_fit_n: int = 500

    escalate_z: float = 3.0
    min_notional_review: float = 0.0

    backend: str = "auto"

    # --- cause attribution percentiles (same rules as Tier 3) -------------
    rev_hi_pct: float = 85.0
    pov_hi_pct: float = 75.0
    passive_lo_pct: float = 15.0
    auction_lo_pct: float = 10.0
    momentum_hi_pct: float = 90.0

    # Tier 3 fields the shared machinery reads. Unused here but kept so the
    # same cost_model / scoring / persist code runs against this config.
    include_interactions: bool = False


CONFIG = Tier4Config()
