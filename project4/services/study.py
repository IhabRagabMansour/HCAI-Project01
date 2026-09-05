"""Study flow: condition counterbalancing and the step state machine.

The participant walks a fixed sequence of steps. Which elicitation condition
comes first depends on the session's counterbalancing group. A within-subject
design counterbalances order to reduce learning and fatigue effects:

    Group AB   pairwise first, then ranking
    Group BA   ranking first, then pairwise

The step sequence is expressed as pure data + pure functions so it can be
unit-tested without Django, and so a view can always answer "which page is this
participant allowed to see right now?" — the server decides, not the URL.
"""

from __future__ import annotations

# ── Conditions ──────────────────────────────────────────────────────────────

PAIRWISE = "pairwise"
RANKING = "ranking"

ORDER_AB = "AB"     # pairwise -> ranking
ORDER_BA = "BA"     # ranking -> pairwise
CONDITION_ORDERS = [ORDER_AB, ORDER_BA]


def assign_condition_order(n_existing_sessions: int) -> str:
    """Alternate AB/BA so the two orders stay balanced as sessions accumulate."""
    return ORDER_AB if n_existing_sessions % 2 == 0 else ORDER_BA


def conditions_for_order(order: str) -> list:
    """The two conditions in presentation order."""
    return [PAIRWISE, RANKING] if order == ORDER_AB else [RANKING, PAIRWISE]


# ── Steps ───────────────────────────────────────────────────────────────────

CONSENT = "consent"
BACKGROUND = "background"
INSTRUCTIONS = "instructions"
PRACTICE = "practice"
CONDITION_1 = "condition_1"
QUESTIONNAIRE_1 = "questionnaire_1"
BREAK = "break"
CONDITION_2 = "condition_2"
QUESTIONNAIRE_2 = "questionnaire_2"
HELDOUT = "heldout"
FINAL = "final"
COMPLETE = "complete"

STEP_SEQUENCE = [
    CONSENT,
    BACKGROUND,
    INSTRUCTIONS,
    PRACTICE,
    CONDITION_1,
    QUESTIONNAIRE_1,
    BREAK,
    CONDITION_2,
    QUESTIONNAIRE_2,
    HELDOUT,
    FINAL,
    COMPLETE,
]

STEP_LABELS = {
    CONSENT: "Information & consent",
    BACKGROUND: "Background questionnaire",
    INSTRUCTIONS: "Instructions",
    PRACTICE: "Practice",
    CONDITION_1: "First elicitation condition",
    QUESTIONNAIRE_1: "Questionnaire",
    BREAK: "Break",
    CONDITION_2: "Second elicitation condition",
    QUESTIONNAIRE_2: "Questionnaire",
    HELDOUT: "Preference check",
    FINAL: "Final comparison",
    COMPLETE: "Done",
}

FIRST_STEP = STEP_SEQUENCE[0]


def next_step(step: str) -> str:
    """The step that follows `step` (COMPLETE is a fixed point)."""
    if step not in STEP_SEQUENCE:
        raise ValueError(f"unknown step: {step!r}")
    index = STEP_SEQUENCE.index(step)
    if index + 1 >= len(STEP_SEQUENCE):
        return COMPLETE
    return STEP_SEQUENCE[index + 1]


def step_number(step: str) -> int:
    """1-based position, for the progress indicator."""
    return STEP_SEQUENCE.index(step) + 1


def total_steps() -> int:
    return len(STEP_SEQUENCE)


def condition_for_step(step: str, order: str) -> str:
    """Which elicitation condition a condition-step runs, given the order."""
    conditions = conditions_for_order(order)
    if step == CONDITION_1:
        return conditions[0]
    if step == CONDITION_2:
        return conditions[1]
    raise ValueError(f"{step!r} is not a condition step")


def is_condition_step(step: str) -> bool:
    return step in (CONDITION_1, CONDITION_2)


# Each per-condition questionnaire asks about the condition just finished.
QUESTIONNAIRE_CONDITION_STEP = {
    QUESTIONNAIRE_1: CONDITION_1,
    QUESTIONNAIRE_2: CONDITION_2,
}


def is_questionnaire_step(step: str) -> bool:
    return step in QUESTIONNAIRE_CONDITION_STEP


def condition_for_questionnaire(step: str, order: str) -> str:
    """Which condition a per-condition questionnaire is asking about."""
    if step not in QUESTIONNAIRE_CONDITION_STEP:
        raise ValueError(f"{step!r} is not a per-condition questionnaire step")
    return condition_for_step(QUESTIONNAIRE_CONDITION_STEP[step], order)


def progress(step: str) -> dict:
    """Progress summary for the participant-facing header."""
    return {
        "step": step,
        "label": STEP_LABELS[step],
        "number": step_number(step),
        "total": total_steps(),
        "percent": round(100.0 * step_number(step) / total_steps()),
    }
