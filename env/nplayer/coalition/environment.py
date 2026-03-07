"""Coalition formation environment wrapping NPlayerEnvironment."""
from __future__ import annotations
from typing import Callable, Optional
from common.games_meta.coalition_config import CoalitionGameConfig, get_coalition_game
from constant_definitions.nplayer.coalition_constants import (
    COALITION_PHASE_NEGOTIATE, COALITION_PHASE_ACTION, ENFORCEMENT_BINDING,
)
from env.nplayer.coalition.models import (
    ActiveCoalition, CoalitionAction, CoalitionObservation,
    CoalitionProposal, CoalitionResponse, CoalitionRoundResult,
)
from env.nplayer.coalition.payoffs import compute_coalition_payoffs
from env.nplayer.coalition.strategies import (
    CoalitionStrategy, CoalitionRandomStrategy, get_coalition_strategy,
)
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.governance.engine import GovernanceEngine
from env.nplayer.governance.models import GovernanceVote
from env.nplayer.models import NPlayerAction, NPlayerObservation

_ONE = int(bool(True))
_ZERO = int()
_ZERO_F = float()


class CoalitionEnvironment:
    """Coalition layer over NPlayerEnvironment with meta-governance."""

    def __init__(self) -> None:
        self._inner = NPlayerEnvironment()
        self._config: Optional[CoalitionGameConfig] = None
        self._strategies: list[CoalitionStrategy] = []
        self._active_coalitions: list[ActiveCoalition] = []
        self._phase: str = ""
        self._coalition_history: list[CoalitionRoundResult] = []
        self._pending_proposals: list[CoalitionProposal] = []
        self._opponent_actions: list[str] = []
        self._score_adjustments: list[float] = []
        self._last_inner_obs: Optional[NPlayerObservation] = None
        self._round_proposals: list[CoalitionProposal] = []
        self._round_responses: list[CoalitionResponse] = []
        self._active_players: set[int] = set()
        self._governance = GovernanceEngine()

    @property
    def active_players(self) -> set[int]: return set(self._active_players)
    @property
    def phase(self) -> str: return self._phase
    @property
    def inner(self) -> NPlayerEnvironment: return self._inner
    @property
    def governance(self) -> GovernanceEngine: return self._governance

    def reset(
        self, game: str, *, coalition_strategies: Optional[list[str]] = None,
        num_rounds: Optional[int] = None, episode_id: Optional[str] = None,
    ) -> CoalitionObservation:
        self._config = get_coalition_game(game)
        n, num_opp = self._config.num_players, self._config.num_players - _ONE
        if coalition_strategies is None:
            self._strategies = [CoalitionRandomStrategy() for _ in range(num_opp)]
        else:
            names = list(coalition_strategies)
            while len(names) < num_opp:
                names.append(names[-_ONE])
            self._strategies = [get_coalition_strategy(s) for s in names]
        self._opponent_actions = [""] * num_opp
        fns = [self._make_fn(i) for i in range(num_opp)]
        self._last_inner_obs = self._inner.reset(
            game, num_rounds=num_rounds, opponent_fns=fns, episode_id=episode_id,
        )
        self._active_coalitions, self._coalition_history = [], []
        self._score_adjustments = [_ZERO_F] * n
        self._active_players = set(range(n))
        self._governance.reset(self._config)
        self._pending_proposals = self._gen_opponent_proposals()
        self._phase = COALITION_PHASE_NEGOTIATE
        return self._build_obs()

    def negotiate_step(self, action: CoalitionAction) -> CoalitionObservation:
        if self._phase != COALITION_PHASE_NEGOTIATE:
            raise RuntimeError("Not in negotiate phase. Call reset() or complete action phase first.")
        assert self._config is not None
        all_proposals = list(self._pending_proposals)
        all_responses: list[CoalitionResponse] = list(action.responses)
        new_coalitions: list[ActiveCoalition] = []
        p_zero_resp = {r.proposal_index: r.accepted for r in action.responses}
        for idx, prop in enumerate(self._pending_proposals):
            if self._proposal_accepted(prop, p_zero_resp, idx):
                new_coalitions.append(ActiveCoalition(
                    members=list(prop.members), agreed_action=prop.agreed_action,
                    side_payment=prop.side_payment))
        for pi, prop in enumerate(action.proposals):
            all_proposals.append(prop)
            if self._primary_proposal_accepted(prop):
                new_coalitions.append(ActiveCoalition(
                    members=list(prop.members), agreed_action=prop.agreed_action,
                    side_payment=prop.side_payment))
        self._active_coalitions = new_coalitions
        self._round_proposals, self._round_responses = all_proposals, all_responses
        self._apply_proposal_targets(all_proposals, new_coalitions)
        self._run_governance(action)
        self._phase = COALITION_PHASE_ACTION
        return self._build_obs()

    def action_step(self, action: NPlayerAction) -> CoalitionObservation:
        if self._phase != COALITION_PHASE_ACTION:
            raise RuntimeError("Not in action phase. Call negotiate_step() first.")
        assert self._config is not None
        n, enforcement = self._config.num_players, self._governance.rules.enforcement
        p_zero_action = action.action
        if enforcement == ENFORCEMENT_BINDING:
            for c in self._active_coalitions:
                if _ZERO in c.members:
                    p_zero_action = c.agreed_action
                    break
        for i, strat in enumerate(self._strategies):
            pidx = i + _ONE
            if pidx not in self._active_players:
                self._opponent_actions[i] = self._config.actions[_ZERO]
                continue
            chosen = strat.choose_action(self._build_obs_for(pidx))
            if enforcement == ENFORCEMENT_BINDING:
                for c in self._active_coalitions:
                    if pidx in c.members:
                        chosen = c.agreed_action
                        break
            self._opponent_actions[i] = chosen
        inner_obs = self._inner.step(NPlayerAction(action=p_zero_action))
        self._last_inner_obs = inner_obs
        base_t = tuple(inner_obs.last_round.payoffs)
        rules = self._governance.rules
        adjusted, defectors, penalties, side_pmts = compute_coalition_payoffs(
            base_t, tuple(inner_obs.last_round.actions), self._active_coalitions,
            rules.enforcement, rules.penalty_numerator, rules.penalty_denominator)
        adj_list = self._governance.apply(list(adjusted), self._active_players)
        for i in range(n):
            if i not in self._active_players:
                adj_list[i] = _ZERO_F
        adjusted = tuple(adj_list)
        for i in range(n):
            self._score_adjustments[i] += adjusted[i] - base_t[i]
        self._coalition_history.append(CoalitionRoundResult(
            round_number=len(self._coalition_history) + _ONE,
            proposals=list(self._round_proposals), responses=list(self._round_responses),
            active_coalitions=list(self._active_coalitions),
            defectors=defectors, penalties=penalties, side_payments=side_pmts))
        if inner_obs.done:
            self._phase = ""
            return self._build_obs(reward_override=adjusted[_ZERO])
        self._active_coalitions = []
        self._pending_proposals = self._gen_opponent_proposals()
        self._phase = COALITION_PHASE_NEGOTIATE
        return self._build_obs(reward_override=adjusted[_ZERO])

    def remove_player(self, player_index: int) -> None:
        """Deactivate a player. Negotiate phase only."""
        if self._phase != COALITION_PHASE_NEGOTIATE:
            raise RuntimeError("Can only remove players during negotiate phase.")
        assert self._config is not None
        if player_index < _ZERO or player_index >= self._config.num_players:
            raise ValueError(f"Player index out of range: {player_index}")
        if player_index not in self._active_players:
            raise ValueError(f"Player {player_index} is already inactive.")
        self._active_players.discard(player_index)

    def add_player(self, player_index: int, strategy: Optional[str] = None) -> None:
        """Reactivate a previously removed player. Negotiate phase only."""
        if self._phase != COALITION_PHASE_NEGOTIATE:
            raise RuntimeError("Can only add players during negotiate phase.")
        assert self._config is not None
        if player_index < _ZERO or player_index >= self._config.num_players:
            raise ValueError(f"Player index out of range: {player_index}")
        if player_index in self._active_players:
            raise ValueError(f"Player {player_index} is already active.")
        self._active_players.add(player_index)
        if strategy is not None and player_index > _ZERO:
            opp_idx = player_index - _ONE
            if opp_idx < len(self._strategies):
                self._strategies[opp_idx] = get_coalition_strategy(strategy)

    def _run_governance(self, action: CoalitionAction) -> None:
        assert self._config is not None
        gov_proposals = list(action.governance_proposals)
        for i, strat in enumerate(self._strategies):
            pidx = i + _ONE
            if pidx in self._active_players and hasattr(strat, "propose_governance"):
                gov_proposals.extend(strat.propose_governance(pidx))
        self._governance.submit_proposals(gov_proposals, self._active_players)
        pending = self._governance.pending_proposals
        votes: list[GovernanceVote] = list(action.governance_votes)
        for i, strat in enumerate(self._strategies):
            pidx = i + _ONE
            if pidx in self._active_players and hasattr(strat, "vote_on_governance"):
                votes.extend(strat.vote_on_governance(pidx, pending))
        self._governance.tally_votes(votes, self._active_players)

    def _apply_proposal_targets(
        self, all_proposals: list[CoalitionProposal], accepted: list[ActiveCoalition],
    ) -> None:
        accepted_members = [tuple(c.members) for c in accepted]
        for prop in all_proposals:
            if tuple(prop.members) not in accepted_members:
                continue
            if prop.exclude_target is not None and prop.exclude_target in self._active_players:
                self._active_players.discard(prop.exclude_target)
            if prop.include_target is not None and prop.include_target not in self._active_players:
                self._active_players.add(prop.include_target)

    def _make_fn(self, idx: int) -> Callable[[NPlayerObservation], NPlayerAction]:
        def fn(obs: NPlayerObservation) -> NPlayerAction:
            return NPlayerAction(action=self._opponent_actions[idx])
        return fn

    def _gen_opponent_proposals(self) -> list[CoalitionProposal]:
        proposals: list[CoalitionProposal] = []
        for i, strat in enumerate(self._strategies):
            pidx = i + _ONE
            if pidx in self._active_players:
                proposals.extend(strat.negotiate(self._build_obs_for(pidx)).proposals)
        return proposals

    def _proposal_accepted(
        self, prop: CoalitionProposal, p_zero_resp: dict[int, bool], idx: int,
    ) -> bool:
        for member in prop.members:
            if member != prop.proposer and member == _ZERO and not p_zero_resp.get(idx, False):
                return False
        return True

    def _primary_proposal_accepted(self, prop: CoalitionProposal) -> bool:
        assert self._config is not None
        for member in prop.members:
            if member == prop.proposer or member == _ZERO:
                continue
            opp_idx = member - _ONE
            if opp_idx < len(self._strategies):
                if not self._strategies[opp_idx].respond_to_proposal(self._build_obs_for(member), prop):
                    return False
            else:
                return False
        return True

    def _build_obs(self, reward_override: Optional[float] = None) -> CoalitionObservation:
        assert self._last_inner_obs is not None and self._config is not None
        base = self._last_inner_obs
        adj_scores = [s + a for s, a in zip(base.scores, self._score_adjustments)]
        reward = reward_override if reward_override is not None else base.reward
        rules = self._governance.rules
        return CoalitionObservation(
            base=base.model_copy(update={"reward": reward}), phase=self._phase,
            active_coalitions=list(self._active_coalitions),
            pending_proposals=list(self._pending_proposals),
            coalition_history=list(self._coalition_history),
            enforcement=rules.enforcement, adjusted_scores=adj_scores,
            active_players=sorted(self._active_players),
            current_rules=rules.model_copy(deep=True),
            pending_governance=self._governance.pending_proposals,
            governance_history=list(rules.governance_history))

    def _build_obs_for(self, player_index: int) -> CoalitionObservation:
        assert self._config is not None
        inner_obs = self._inner._build_observation(player_index)
        adj_scores = [s + a for s, a in zip(inner_obs.scores, self._score_adjustments)]
        rules = self._governance.rules
        return CoalitionObservation(
            base=inner_obs, phase=self._phase,
            active_coalitions=list(self._active_coalitions),
            pending_proposals=list(self._pending_proposals),
            coalition_history=list(self._coalition_history),
            enforcement=rules.enforcement, adjusted_scores=adj_scores,
            active_players=sorted(self._active_players),
            current_rules=rules.model_copy(deep=True),
            pending_governance=self._governance.pending_proposals,
            governance_history=list(rules.governance_history))
