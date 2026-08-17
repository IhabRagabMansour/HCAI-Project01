import numpy as np
from django.shortcuts import get_object_or_404, redirect, render

from .models import QuestionnaireResponse, StudySession
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
}


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
