from __future__ import annotations

from dataclasses import dataclass

from engine.probability import MarketState, normalize_percentages


@dataclass(frozen=True)
class DecisionResult:
    decision_probabilities: dict[str, float]
    final_decision: str
    confidence: int
    market_bias: str


def build_decision_probabilities(next_probs: dict[str, float], state: MarketState) -> dict[str, float]:
    """Separate trading decision scoring. BUY is not treated as a direct alias of UP."""
    scores = {
        "BUY": 12.0,
        "SELL": 12.0,
        "HOLD": 24.0,
    }

    if state.is_sideways:
        scores["HOLD"] += 70.0
        scores["BUY"] *= 0.35
        scores["SELL"] *= 0.35
        return normalize_percentages(scores)

    scores["HOLD"] += next_probs["SIDEWAYS"] * 0.85

    if state.bullish_trend:
        scores["BUY"] += 16.0
        scores["SELL"] -= 5.0
    if state.bearish_trend:
        scores["SELL"] += 16.0
        scores["BUY"] -= 5.0

    if state.bullish_rsi:
        scores["BUY"] += 12.0
    elif state.bearish_rsi:
        scores["SELL"] += 12.0
    else:
        scores["HOLD"] += 12.0

    if next_probs["UP"] >= 58 and state.bullish_trend and state.bullish_rsi:
        scores["BUY"] += next_probs["UP"] * 0.35
    if next_probs["DOWN"] >= 58 and state.bearish_trend and state.bearish_rsi:
        scores["SELL"] += next_probs["DOWN"] * 0.35

    if state.volume_spike:
        if scores["BUY"] > scores["SELL"]:
            scores["BUY"] += 8.0
        elif scores["SELL"] > scores["BUY"]:
            scores["SELL"] += 8.0

    return normalize_percentages(scores)


def decide(next_probs: dict[str, float], state: MarketState) -> DecisionResult:
    decision_probs = build_decision_probabilities(next_probs, state)

    if state.is_sideways or decision_probs["HOLD"] >= 45:
        final_decision = "HOLD"
    elif decision_probs["BUY"] >= 52 and decision_probs["BUY"] - decision_probs["SELL"] >= 8:
        final_decision = "BUY"
    elif decision_probs["SELL"] >= 52 and decision_probs["SELL"] - decision_probs["BUY"] >= 8:
        final_decision = "SELL"
    else:
        final_decision = "HOLD"

    confidence = score_confidence(final_decision, next_probs, decision_probs, state)
    return DecisionResult(
        decision_probabilities=decision_probs,
        final_decision=final_decision,
        confidence=confidence,
        market_bias=state.bias,
    )


def score_confidence(
    final_decision: str,
    next_probs: dict[str, float],
    decision_probs: dict[str, float],
    state: MarketState,
) -> int:
    if final_decision == "HOLD":
        model_component = next_probs["SIDEWAYS"]
    else:
        model_component = max(next_probs["UP"], next_probs["DOWN"])

    confidence = decision_probs[final_decision] * 0.70 + model_component * 0.30
    if state.volume_spike and final_decision != "HOLD":
        confidence += 5
    if state.is_sideways and final_decision == "HOLD":
        confidence += 8
    return int(max(0, min(round(confidence), 100)))
