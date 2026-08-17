"""Anonymised CSV exports of the collected study data.

Two shapes, because the planned analysis needs two:

    sessions.csv   one row per participant — the paired, analysis-ready format
                   for the primary within-subject comparison
    trials.csv     one row per task — the interaction-level data behind the
                   time and effort measures

Neither carries a name, an email or any other direct identifier: the only key is
the pseudonymous participant code. In a real deployment these endpoints would sit
behind researcher authentication, as the report's ethics section states.
"""

from __future__ import annotations

import csv
import io

from ..models import (
    PairwiseTrial, PreferenceModelFit, QuestionnaireResponse, RankingTrial,
    StudySession,
)
from .study import PAIRWISE, RANKING

# Subjective items, in the order they are asked.
CONDITION_ITEMS = ["ease", "effort", "expressive", "confidence", "willing"]
BACKGROUND_ITEMS = ["age_range", "movie_frequency", "recommender_familiarity"]
FINAL_ITEMS = ["preferred", "expressive", "effort"]

SESSION_COLUMNS = (
    ["participant_code", "condition_order", "consented", "completed",
     "started_at", "completed_at"]
    + [f"background_{item}" for item in BACKGROUND_ITEMS]
    + ["n_pairwise_trials", "n_rankings", "n_heldout",
       "pairwise_median_rt_ms", "ranking_median_rt_ms",
       "pairwise_total_time_ms", "ranking_total_time_ms",
       "pairwise_heldout_accuracy", "pairwise_heldout_log_loss",
       "ranking_heldout_accuracy", "ranking_heldout_log_loss"]
    + [f"pairwise_{item}" for item in CONDITION_ITEMS]
    + [f"ranking_{item}" for item in CONDITION_ITEMS]
    + [f"final_{item}" for item in FINAL_ITEMS]
    + ["pairwise_comment", "ranking_comment", "final_comment"]
)

TRIAL_COLUMNS = [
    "participant_code", "condition_order", "interface", "block", "task_index",
    "movies_shown", "response", "response_time_ms", "created_at",
]


def _median(values):
    """Median of the non-null values, or '' when there are none."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return ""
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _total(values):
    known = [v for v in values if v is not None]
    return sum(known) if known else ""


def _stamp(value):
    return value.isoformat() if value else ""


def _questionnaires(session):
    """This session's questionnaires, keyed for lookup."""
    found = {}
    for answer in QuestionnaireResponse.objects.filter(session=session):
        key = answer.condition if answer.kind == QuestionnaireResponse.KIND_CONDITION \
            else answer.kind
        found[key] = answer.answers or {}
    return found


def _session_row(session) -> list:
    pairwise = list(PairwiseTrial.objects.filter(session=session, block="main"))
    rankings = list(RankingTrial.objects.filter(session=session, block="main"))
    heldout = list(PairwiseTrial.objects.filter(session=session, block="heldout"))

    answers = _questionnaires(session)
    background = answers.get(QuestionnaireResponse.KIND_BACKGROUND, {})
    final = answers.get(QuestionnaireResponse.KIND_FINAL, {})
    fits = {fit.condition: fit for fit in PreferenceModelFit.objects.filter(session=session)}

    def fit_value(condition, attribute):
        fit = fits.get(condition)
        value = getattr(fit, attribute) if fit else None
        return "" if value is None else value

    row = [
        session.participant_code, session.condition_order,
        session.consented, session.is_complete,
        _stamp(session.started_at), _stamp(session.completed_at),
    ]
    row += [background.get(item, "") for item in BACKGROUND_ITEMS]
    row += [
        len(pairwise), len(rankings), len(heldout),
        _median(t.response_time_ms for t in pairwise),
        _median(t.response_time_ms for t in rankings),
        _total(t.response_time_ms for t in pairwise),
        _total(t.response_time_ms for t in rankings),
        fit_value(PAIRWISE, "heldout_accuracy"),
        fit_value(PAIRWISE, "heldout_log_loss"),
        fit_value(RANKING, "heldout_accuracy"),
        fit_value(RANKING, "heldout_log_loss"),
    ]
    for condition in (PAIRWISE, RANKING):
        given = answers.get(condition, {})
        row += [given.get(item, "") for item in CONDITION_ITEMS]
    row += [final.get(item, "") for item in FINAL_ITEMS]
    row += [
        answers.get(PAIRWISE, {}).get("comment", ""),
        answers.get(RANKING, {}).get("comment", ""),
        final.get("comment", ""),
    ]
    return row


def _trial_rows(session):
    for trial in PairwiseTrial.objects.filter(session=session):
        yield [
            session.participant_code, session.condition_order, PAIRWISE,
            trial.block, trial.task_index,
            f"{trial.left_movie_id};{trial.right_movie_id}",
            trial.chosen_movie_id,
            "" if trial.response_time_ms is None else trial.response_time_ms,
            _stamp(trial.created_at),
        ]
    for trial in RankingTrial.objects.filter(session=session):
        yield [
            session.participant_code, session.condition_order, RANKING,
            trial.block, trial.task_index,
            ";".join(str(m) for m in trial.movie_ids),
            ";".join(str(m) for m in trial.ranked_movie_ids),
            "" if trial.response_time_ms is None else trial.response_time_ms,
            _stamp(trial.created_at),
        ]


def _render(header, rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def sessions_csv() -> str:
    """One row per participant: the paired format the primary analysis needs."""
    sessions = StudySession.objects.all().order_by("started_at")
    return _render(SESSION_COLUMNS, (_session_row(s) for s in sessions))


def trials_csv() -> str:
    """One row per task, across every block and both interfaces."""
    rows = []
    for session in StudySession.objects.all().order_by("started_at"):
        rows.extend(_trial_rows(session))
    return _render(TRIAL_COLUMNS, rows)
