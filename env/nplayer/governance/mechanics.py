"""Pure payoff modifier functions for governance mechanics."""

from __future__ import annotations

from constant_definitions.nplayer.governance_constants import (
    MECHANIC_TAXATION,
    MECHANIC_REDISTRIBUTION,
    MECHANIC_INSURANCE,
    MECHANIC_QUOTA,
    MECHANIC_SUBSIDY,
    MECHANIC_VETO,
    MECHANIC_ORDER,
    REDISTRIBUTION_PROPORTIONAL,
)
from env.nplayer.governance.models import MechanicConfig, RuntimeRules

_ZERO = int()
_ONE = int(bool(True))
_ZERO_F = float()


def _apply_taxation(
    payoffs: list[float], active: set[int], cfg: MechanicConfig,
) -> list[float]:
    """Active players pay tax_rate * payoff into a pool, distributed equally."""
    n_active = len(active)
    if n_active == _ZERO:
        return payoffs
    rate = cfg.tax_rate_numerator / cfg.tax_rate_denominator
    pool = _ZERO_F
    for i in active:
        pool += payoffs[i] * rate
    share = pool / n_active
    result = list(payoffs)
    for i in active:
        result[i] = result[i] - payoffs[i] * rate + share
    return result


def _apply_redistribution(
    payoffs: list[float], active: set[int], cfg: MechanicConfig,
) -> list[float]:
    """Equal mode: everyone gets mean. Proportional: dampen toward mean."""
    n_active = len(active)
    if n_active == _ZERO:
        return payoffs
    total = sum(payoffs[i] for i in active)
    mean = total / n_active
    result = list(payoffs)
    if cfg.redistribution_mode == REDISTRIBUTION_PROPORTIONAL:
        damping = cfg.damping_numerator / cfg.damping_denominator
        for i in active:
            result[i] = result[i] + damping * (mean - result[i])
    else:
        for i in active:
            result[i] = mean
    return result


def _apply_insurance(
    payoffs: list[float], active: set[int], cfg: MechanicConfig,
) -> list[float]:
    """All contribute a fraction; below-threshold players receive payout."""
    n_active = len(active)
    if n_active == _ZERO:
        return payoffs
    contrib_rate = cfg.insurance_contribution_numerator / cfg.insurance_contribution_denominator
    pool = _ZERO_F
    result = list(payoffs)
    for i in active:
        contrib = result[i] * contrib_rate
        pool += contrib
        result[i] -= contrib
    mean_pre = sum(payoffs[i] for i in active) / n_active
    threshold = mean_pre * cfg.insurance_threshold_numerator / cfg.insurance_threshold_denominator
    claimants = [i for i in active if payoffs[i] < threshold]
    if claimants:
        payout = pool / len(claimants)
        for i in claimants:
            result[i] += payout
    return result


def _apply_quota(
    payoffs: list[float], active: set[int], cfg: MechanicConfig,
) -> list[float]:
    """Cap individual payoff at maximum; excess redistributed to below-cap."""
    cap = cfg.quota_max
    result = list(payoffs)
    excess = _ZERO_F
    below_cap = []
    for i in active:
        if result[i] > cap:
            excess += result[i] - cap
            result[i] = cap
        else:
            below_cap.append(i)
    if below_cap and excess > _ZERO_F:
        share = excess / len(below_cap)
        for i in below_cap:
            result[i] += share
    return result


def _apply_subsidy(
    payoffs: list[float], active: set[int], cfg: MechanicConfig,
) -> list[float]:
    """Floor on payoffs, funded by fraction from above-floor players."""
    floor_val = cfg.subsidy_floor
    fund_rate = cfg.subsidy_fund_rate_numerator / cfg.subsidy_fund_rate_denominator
    result = list(payoffs)
    # Collect funds from above-floor players
    pool = _ZERO_F
    for i in active:
        if result[i] > floor_val:
            contrib = (result[i] - floor_val) * fund_rate
            pool += contrib
            result[i] -= contrib
    # Distribute to below-floor players
    below = [i for i in active if payoffs[i] < floor_val]
    if below and pool > _ZERO_F:
        need_total = sum(floor_val - payoffs[i] for i in below)
        for i in below:
            need = floor_val - payoffs[i]
            if need_total > _ZERO_F:
                result[i] += min(need, pool * need / need_total)
    return result


def _apply_veto(
    payoffs: list[float], active: set[int], cfg: MechanicConfig,
) -> list[float]:
    """Designated player triggers equalization if their payoff falls below mean."""
    vp = cfg.veto_player
    if vp not in active:
        return payoffs
    n_active = len(active)
    if n_active == _ZERO:
        return payoffs
    total = sum(payoffs[i] for i in active)
    mean = total / n_active
    if payoffs[vp] < mean:
        result = list(payoffs)
        for i in active:
            result[i] = mean
        return result
    return payoffs


_MECHANIC_FNS = {
    MECHANIC_TAXATION: _apply_taxation,
    MECHANIC_REDISTRIBUTION: _apply_redistribution,
    MECHANIC_INSURANCE: _apply_insurance,
    MECHANIC_QUOTA: _apply_quota,
    MECHANIC_SUBSIDY: _apply_subsidy,
    MECHANIC_VETO: _apply_veto,
}


def apply_mechanics(
    payoffs: list[float], rules: RuntimeRules, active_players: set[int],
) -> list[float]:
    """Run all enabled mechanics in fixed order."""
    result = list(payoffs)
    for name in MECHANIC_ORDER:
        if rules.mechanics.get(name, False):
            fn = _MECHANIC_FNS.get(name)
            if fn is not None:
                result = fn(result, active_players, rules.mechanic_config)
    return result
