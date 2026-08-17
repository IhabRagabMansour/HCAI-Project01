import numpy as np
from django.shortcuts import get_object_or_404, redirect, render

from .models import (
    BLOCK_HELDOUT, BLOCK_MAIN, BLOCK_PRACTICE,
    PairwiseTrial, QuestionnaireResponse, RankingTrial, StudySession,
)
from .services import study as study_flow
from .services.data import get_movie_corpus, genre_vocabulary
from .services.features import describe_feature_groups, get_features
from .services.sampling import DEFAULT_CONFIG, build_trial_plan, new_session_seed

SESSION_KEY = "project4_session_pk"

# Which URL serves each step. Steps not listed yet are served by the
# placeholder view until their stage lands.
STEP_URL_NAMES = {
    study_flow.CONSENT: "project4:consent",
    study_flow.BACKGROUND: "project4:background",
    study_flow.INSTRUCTIONS: "project4:instructions",
    study_flow.PRACTICE: "project4:practice",
    study_flow.CONDITION_1: "project4:condition",
    study_flow.CONDITION_2: "project4:condition",
}


def _plan_for(session):
    """Regenerate the session's deterministic trial plan."""
    return build_trial_plan(session.seed, get_movie_corpus().n_movies, DEFAULT_CONFIG)


def _parse_response_time(request):
    """Client-reported response time in ms, ignored if malformed."""
    try:
        value = int(request.POST.get("response_time_ms", ""))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value < 1000 * 60 * 60 else None


def _current_session(request):
    """The StudySession attached to this browser session, if any."""
    pk = request.session.get(SESSION_KEY)
    if not pk:
        return None
    return StudySession.objects.filter(pk=pk).first()


def _require_session(request):
    """Return the session, or None if the participant has not started."""
    return _current_session(request)


def _advance(session, from_step):
    """Move the session to the next step (only if it is still on `from_step`)."""
    if session.current_step == from_step:
        session.current_step = study_flow.next_step(from_step)
        session.save(update_fields=["current_step"])
    return session


def index(request):
    """Landing page: project description + (later) report download and study start."""
    corpus = get_movie_corpus()
    df = corpus.df
    genres = genre_vocabulary(df)

    context = {
        "title": "Project 4 — Preference Elicitation",
        "n_movies": corpus.n_movies,
        "n_genres": len(genres),
        "genres": genres,
        "year_min": int(df["title_year"].min()),
        "year_max": int(df["title_year"].max()),
        "sample_movies": corpus.display_records(range(6)),
    }
    return render(request, "project4/index.html", context)


# Movies used to illustrate the feature representation on the features page.
_EXAMPLE_MOVIE_IDS = [0, 1, 2]


def features(request):
    """Task 1: the movie feature representation and how it is extracted."""
    corpus = get_movie_corpus()
    X, encoder = get_features()

    examples = []
    for movie_id in _EXAMPLE_MOVIE_IDS:
        record = corpus.display_record(movie_id)
        vector = X[movie_id]
        record["nonzero"] = [
            {"name": encoder.feature_names[i], "value": float(vector[i])}
            for i in np.nonzero(vector)[0]
        ]
        examples.append(record)

    context = {
        "title": "Task 1 — Feature Representation",
        "n_movies": corpus.n_movies,
        "dim": encoder.dim,
        "groups": describe_feature_groups(encoder),
        "feature_names": encoder.feature_names,
        "examples": examples,
    }
    return render(request, "project4/features.html", context)


# ── Study flow ──────────────────────────────────────────────────────────────

def start_study(request):
    """Create a fresh participant session and enter the study."""
    if request.method != "POST":
        return redirect("project4:index")

    order = study_flow.assign_condition_order(StudySession.objects.count())
    session = StudySession.objects.create(
        condition_order=order,
        seed=new_session_seed(),
        current_step=study_flow.FIRST_STEP,
    )
    request.session[SESSION_KEY] = session.pk
    return redirect("project4:study")


def study(request):
    """Router: send the participant to whichever step they are actually on."""
    session = _require_session(request)
    if session is None:
        return redirect("project4:index")

    url_name = STEP_URL_NAMES.get(session.current_step)
    if url_name is None:
        return render(request, "project4/pending.html", {
            "title": "Study",
            "progress": study_flow.progress(session.current_step),
            "session": session,
        })
    return redirect(url_name)


def _step_view(request, step, template, context=None, on_post=None):
    """Shared plumbing for a linear study step.

    Guards that the participant is really on this step, renders the template on
    GET, runs `on_post` and advances on POST.
    """
    session = _require_session(request)
    if session is None:
        return redirect("project4:index")
    if session.current_step != step:
        return redirect("project4:study")

    if request.method == "POST":
        if on_post is not None:
            on_post(request, session)
        _advance(session, step)
        return redirect("project4:study")

    data = {
        "session": session,
        "progress": study_flow.progress(step),
    }
    data.update(context or {})
    return render(request, template, data)


def consent(request):
    """Step 1 — information sheet and informed consent."""
    def handle(request, session):
        session.consented = True
        session.save(update_fields=["consented"])

    return _step_view(
        request, study_flow.CONSENT, "project4/consent.html",
        context={"title": "Information & Consent", "config": DEFAULT_CONFIG},
        on_post=handle,
    )


BACKGROUND_FIELDS = ["age_range", "movie_frequency", "recommender_familiarity"]


def background(request):
    """Step 2 — short background questionnaire (no identifying data)."""
    def handle(request, session):
        QuestionnaireResponse.objects.create(
            session=session,
            kind=QuestionnaireResponse.KIND_BACKGROUND,
            answers={f: request.POST.get(f, "") for f in BACKGROUND_FIELDS},
        )

    return _step_view(
        request, study_flow.BACKGROUND, "project4/background.html",
        context={"title": "About You"},
        on_post=handle,
    )


def instructions(request):
    """Step 3 — task instructions before the practice trials."""
    session = _current_session(request)
    order = session.condition_order if session else study_flow.ORDER_AB
    first_condition = study_flow.conditions_for_order(order)[0]

    return _step_view(
        request, study_flow.INSTRUCTIONS, "project4/instructions.html",
        context={
            "title": "Instructions",
            "config": DEFAULT_CONFIG,
            "first_condition": first_condition,
        },
    )


# ── Trial engines ───────────────────────────────────────────────────────────
#
# One engine per elicitation interface, each serving every block that uses it:
# practice, the main condition, and (for pairwise) the shared held-out check.
# The trial index is simply how many trials of that block are already stored, so
# a refresh can never double-count and a reload resumes exactly where the
# participant left off.
#
# A runner returns None once its block is finished. The caller decides what
# happens next, which is what lets the practice step chain two blocks together.

def _pairwise_done_count(session, block) -> int:
    return PairwiseTrial.objects.filter(session=session, block=block).count()


def _record_pairwise(request, session, block, index, pair) -> bool:
    """Validate and store one pairwise choice. Returns True if it was recorded."""
    left_id, right_id = pair
    try:
        chosen = int(request.POST.get("chosen_movie_id", ""))
    except (TypeError, ValueError):
        return False
    if chosen not in (left_id, right_id):
        return False

    PairwiseTrial.objects.update_or_create(
        session=session, block=block, task_index=index,
        defaults={
            "left_movie_id": left_id,
            "right_movie_id": right_id,
            "chosen_movie_id": chosen,
            "response_time_ms": _parse_response_time(request),
        },
    )
    return True


def _run_pairwise_block(request, session, block, pairs, *, step, title, is_practice=False):
    """One trial of a pairwise block, or None once the block is finished."""
    index = _pairwise_done_count(session, block)
    if index >= len(pairs):
        return None

    if request.method == "POST":
        # An invalid choice simply is not recorded, so the same trial reappears.
        _record_pairwise(request, session, block, index, pairs[index])
        return redirect("project4:study")

    corpus = get_movie_corpus()
    left_id, right_id = pairs[index]
    return render(request, "project4/pairwise.html", {
        "title": title,
        "progress": study_flow.progress(step),
        "session": session,
        "movies": [corpus.display_record(left_id), corpus.display_record(right_id)],
        "trial_number": index + 1,
        "trial_total": len(pairs),
        "is_practice": is_practice,
    })


# ── Ranking trial engine ────────────────────────────────────────────────────
#
# The browser posts the whole order back as a comma-separated list, which the
# server accepts only if it is exactly a permutation of the movies it showed —
# never trusting the client to have kept the set intact.

def _ranking_done_count(session, block) -> int:
    return RankingTrial.objects.filter(session=session, block=block).count()


def _parse_submitted_order(request, movie_ids):
    """The posted order, or None unless it permutes exactly the shown movies."""
    raw = request.POST.get("ranked_movie_ids", "")
    try:
        order = [int(value) for value in raw.split(",") if value.strip()]
    except (TypeError, ValueError):
        return None
    # One comparison rejects duplicates, omissions, extras and foreign ids.
    if sorted(order) != sorted(int(m) for m in movie_ids):
        return None
    return order


def _apply_move(order, move):
    """Apply one `up:<id>` / `down:<id>` swap — the no-JavaScript fallback."""
    direction, _, raw = move.partition(":")
    try:
        movie_id = int(raw)
    except (TypeError, ValueError):
        return order
    if movie_id not in order:
        return order

    source = order.index(movie_id)
    target = source - 1 if direction == "up" else source + 1
    if direction not in ("up", "down") or not 0 <= target < len(order):
        return order

    order = list(order)
    order[source], order[target] = order[target], order[source]
    return order


def _run_ranking_block(request, session, block, groups, *, step, title, is_practice=False):
    """One trial of a ranking block, or None once the block is finished."""
    index = _ranking_done_count(session, block)
    if index >= len(groups):
        return None

    movie_ids = groups[index]
    order = list(movie_ids)

    if request.method == "POST":
        submitted = _parse_submitted_order(request, movie_ids)
        move = request.POST.get("move", "")
        if move:
            # Fallback path: one move per request, then re-render. Never reached
            # when JavaScript is available, which reorders in place instead.
            order = _apply_move(submitted if submitted is not None else order, move)
        elif submitted is not None:
            RankingTrial.objects.update_or_create(
                session=session, block=block, task_index=index,
                defaults={
                    "movie_ids": [int(m) for m in movie_ids],
                    "ranked_movie_ids": submitted,
                    "response_time_ms": _parse_response_time(request),
                },
            )
            return redirect("project4:study")
        else:
            # A tampered or truncated order is discarded rather than stored
            # half-right; the participant is shown the same task again.
            return redirect("project4:study")

    corpus = get_movie_corpus()
    return render(request, "project4/ranking.html", {
        "title": title,
        "progress": study_flow.progress(step),
        "session": session,
        "movies": [corpus.display_record(movie_id) for movie_id in order],
        "order_csv": ",".join(str(movie_id) for movie_id in order),
        "trial_number": index + 1,
        "trial_total": len(groups),
        "is_practice": is_practice,
    })


def _run_condition_block(request, session, which, plan, block, *, step, is_practice=False):
    """Dispatch to whichever interface `which` names, for the given block."""
    if which == study_flow.PAIRWISE:
        return _run_pairwise_block(
            request, session, block,
            plan.practice_pairwise if is_practice else plan.pairwise,
            step=step, is_practice=is_practice,
            title="Practice" if is_practice else "Which would you rather watch?",
        )
    return _run_ranking_block(
        request, session, block,
        plan.practice_ranking if is_practice else plan.ranking,
        step=step, is_practice=is_practice,
        title="Practice" if is_practice else "Put these films in order",
    )


def practice(request):
    """Step 4 — unscored practice with *both* interfaces, before any measurement.

    Both are practised up front, in the order the participant will meet them, so
    neither condition gets a warm-up the other did not.
    """
    session = _require_session(request)
    if session is None:
        return redirect("project4:index")
    if session.current_step != study_flow.PRACTICE:
        return redirect("project4:study")

    plan = _plan_for(session)
    for which in study_flow.conditions_for_order(session.condition_order):
        response = _run_condition_block(
            request, session, which, plan, BLOCK_PRACTICE,
            step=study_flow.PRACTICE, is_practice=True,
        )
        if response is not None:
            return response

    _advance(session, study_flow.PRACTICE)
    return redirect("project4:study")


def condition(request):
    """Steps 5 & 8 — whichever elicitation condition this session runs now."""
    session = _require_session(request)
    if session is None:
        return redirect("project4:index")
    step = session.current_step
    if not study_flow.is_condition_step(step):
        return redirect("project4:study")

    which = study_flow.condition_for_step(step, session.condition_order)
    response = _run_condition_block(
        request, session, which, _plan_for(session), BLOCK_MAIN, step=step,
    )
    if response is not None:
        return response

    _advance(session, step)
    return redirect("project4:study")
