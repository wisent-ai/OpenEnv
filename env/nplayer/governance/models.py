"""Data models for the meta-governance system."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from constant_definitions.game_constants import DEFAULT_NONE
from constant_definitions.nplayer.coalition_constants import (
    ENFORCEMENT_CHEAP_TALK,
    COALITION_DEFAULT_PENALTY_NUMERATOR,
    COALITION_DEFAULT_PENALTY_DENOMINATOR,
)
from constant_definitions.nplayer.governance_constants import (
    GOVERNANCE_PROPOSAL_PARAMETER,
    GOVERNANCE_DEFAULT_TAX_RATE_NUMERATOR,
    GOVERNANCE_DEFAULT_TAX_RATE_DENOMINATOR,
    GOVERNANCE_DEFAULT_REDISTRIBUTION_MODE,
    GOVERNANCE_DEFAULT_REDISTRIBUTION_DAMPING_NUMERATOR,
    GOVERNANCE_DEFAULT_REDISTRIBUTION_DAMPING_DENOMINATOR,
    GOVERNANCE_DEFAULT_INSURANCE_CONTRIBUTION_NUMERATOR,
    GOVERNANCE_DEFAULT_INSURANCE_CONTRIBUTION_DENOMINATOR,
    GOVERNANCE_DEFAULT_INSURANCE_THRESHOLD_NUMERATOR,
    GOVERNANCE_DEFAULT_INSURANCE_THRESHOLD_DENOMINATOR,
    GOVERNANCE_DEFAULT_QUOTA_MAX,
    GOVERNANCE_DEFAULT_SUBSIDY_FLOOR,
    GOVERNANCE_DEFAULT_SUBSIDY_FUND_RATE_NUMERATOR,
    GOVERNANCE_DEFAULT_SUBSIDY_FUND_RATE_DENOMINATOR,
    GOVERNANCE_DEFAULT_VETO_PLAYER,
)

_ZERO = int()
_ONE = int(bool(True))
_ZERO_F = float()


class MechanicConfig(BaseModel):
    """Per-mechanic parameter bundle."""

    # taxation
    tax_rate_numerator: int = Field(default=GOVERNANCE_DEFAULT_TAX_RATE_NUMERATOR)
    tax_rate_denominator: int = Field(default=GOVERNANCE_DEFAULT_TAX_RATE_DENOMINATOR)

    # redistribution
    redistribution_mode: str = Field(default=GOVERNANCE_DEFAULT_REDISTRIBUTION_MODE)
    damping_numerator: int = Field(default=GOVERNANCE_DEFAULT_REDISTRIBUTION_DAMPING_NUMERATOR)
    damping_denominator: int = Field(default=GOVERNANCE_DEFAULT_REDISTRIBUTION_DAMPING_DENOMINATOR)

    # insurance
    insurance_contribution_numerator: int = Field(
        default=GOVERNANCE_DEFAULT_INSURANCE_CONTRIBUTION_NUMERATOR,
    )
    insurance_contribution_denominator: int = Field(
        default=GOVERNANCE_DEFAULT_INSURANCE_CONTRIBUTION_DENOMINATOR,
    )
    insurance_threshold_numerator: int = Field(
        default=GOVERNANCE_DEFAULT_INSURANCE_THRESHOLD_NUMERATOR,
    )
    insurance_threshold_denominator: int = Field(
        default=GOVERNANCE_DEFAULT_INSURANCE_THRESHOLD_DENOMINATOR,
    )

    # quota
    quota_max: float = Field(default=float(GOVERNANCE_DEFAULT_QUOTA_MAX))

    # subsidy
    subsidy_floor: float = Field(default=float(GOVERNANCE_DEFAULT_SUBSIDY_FLOOR))
    subsidy_fund_rate_numerator: int = Field(
        default=GOVERNANCE_DEFAULT_SUBSIDY_FUND_RATE_NUMERATOR,
    )
    subsidy_fund_rate_denominator: int = Field(
        default=GOVERNANCE_DEFAULT_SUBSIDY_FUND_RATE_DENOMINATOR,
    )

    # veto
    veto_player: int = Field(default=GOVERNANCE_DEFAULT_VETO_PLAYER)


class RuntimeRules(BaseModel):
    """Mutable overlay on top of frozen CoalitionGameConfig."""

    enforcement: str = Field(default=ENFORCEMENT_CHEAP_TALK)
    penalty_numerator: int = Field(default=COALITION_DEFAULT_PENALTY_NUMERATOR)
    penalty_denominator: int = Field(default=COALITION_DEFAULT_PENALTY_DENOMINATOR)
    allow_side_payments: bool = Field(default=False)

    mechanics: dict[str, bool] = Field(
        default_factory=dict,
        description="Mechanic name -> active flag",
    )
    mechanic_config: MechanicConfig = Field(default_factory=MechanicConfig)

    custom_modifier_keys: list[str] = Field(
        default_factory=list,
        description="Keys of active custom modifiers",
    )

    governance_history: list[GovernanceResult] = Field(default_factory=list)


class GovernanceProposal(BaseModel):
    """A governance change proposed by a player."""

    proposer: int = Field(..., description="Player index of the proposer")
    proposal_type: str = Field(
        default=GOVERNANCE_PROPOSAL_PARAMETER,
        description="One of: parameter, mechanic, custom",
    )

    # parameter changes
    parameter_name: Optional[str] = Field(
        default=DEFAULT_NONE,
        description="Name of the parameter to change (enforcement, penalty_numerator, etc.)",
    )
    parameter_value: Optional[Any] = Field(
        default=DEFAULT_NONE,
        description="New value for the parameter",
    )

    # mechanic toggles
    mechanic_name: Optional[str] = Field(
        default=DEFAULT_NONE,
        description="Mechanic to activate/deactivate",
    )
    mechanic_active: Optional[bool] = Field(
        default=DEFAULT_NONE,
        description="True to activate, False to deactivate",
    )
    mechanic_params: Optional[dict[str, Any]] = Field(
        default=DEFAULT_NONE,
        description="Optional parameter overrides for the mechanic",
    )

    # custom modifiers
    custom_modifier_key: Optional[str] = Field(
        default=DEFAULT_NONE,
        description="Key of the custom modifier to activate/deactivate",
    )
    custom_modifier_active: Optional[bool] = Field(
        default=DEFAULT_NONE,
        description="True to activate, False to deactivate",
    )


class GovernanceVote(BaseModel):
    """A player's vote on a governance proposal."""

    voter: int = Field(..., description="Player index of the voter")
    proposal_index: int = Field(..., description="Index into the proposals list")
    approve: bool = Field(..., description="Whether the voter approves")


class GovernanceResult(BaseModel):
    """Record of governance activity for one round."""

    proposals: list[GovernanceProposal] = Field(default_factory=list)
    votes: list[GovernanceVote] = Field(default_factory=list)
    adopted: list[int] = Field(
        default_factory=list,
        description="Indices of adopted proposals",
    )
    rejected: list[int] = Field(
        default_factory=list,
        description="Indices of rejected proposals",
    )
    rules_snapshot: Optional[RuntimeRules] = Field(
        default=DEFAULT_NONE,
        description="Rules state after this round of governance",
    )


# Allow RuntimeRules to reference GovernanceResult (forward ref)
RuntimeRules.model_rebuild()
