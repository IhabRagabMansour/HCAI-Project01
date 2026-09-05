"""Generate the mandatory PDF report for Project 4.

We organize the report into fourteen sections covering:
introduction, dataset, Task 1 features, the pairwise model, Task 2 ranking
model, research question, experimental design, participants, procedure,
measures, planned analysis, ethics, interface, and limitations.

Every number describing the corpus, the feature space or the elicitation budget
is read from the live services rather than typed in, so the report cannot drift
away from the implementation. The study has *not* been run, so the report
contains no participant results — only the protocol that would produce them.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

from .data import genre_vocabulary, get_movie_corpus
from .features import (
    NUMERIC_COLUMNS, RATING_BUCKETS, describe_feature_groups, get_features,
)
from .preference_models import DEFAULT_REGULARIZATION
from .sampling import DEFAULT_CONFIG

# Sample-size planning constants (see section 8).
ALPHA = 0.05
POWER = 0.80
TARGET_EFFECT = 0.5          # Cohen's d for the paired difference, medium
PLANNED_N = 34
RECRUIT_N = 40


def _styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=14, spaceBefore=10,
                        spaceAfter=6, textColor=colors.HexColor("#275CB2"))
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=11.5, spaceBefore=9,
                        spaceAfter=4, textColor=colors.HexColor("#1e4a9a"))
    body = ParagraphStyle("Body", parent=base["BodyText"], fontSize=9.7, leading=13.6,
                          alignment=TA_LEFT, spaceAfter=6)
    # Formulas use plain-text mathematical notation for reliable PDF rendering.
    # materials. The built-in PDF fonts are WinAnsi-encoded and carry no Greek or
    # mathematical operators, so a typeset sigma or product sign would be dropped
    # silently; ASCII renders identically everywhere and cannot fail that way.
    formula = ParagraphStyle("Formula", parent=body, fontName="Courier",
                             fontSize=9, leading=13, leftIndent=16,
                             spaceBefore=3, spaceAfter=8,
                             textColor=colors.HexColor("#1e4a9a"))
    callout = ParagraphStyle("Callout", parent=body, fontName="Helvetica-Oblique",
                             leftIndent=16, spaceBefore=3, spaceAfter=7,
                             textColor=colors.HexColor("#333333"))
    title = ParagraphStyle("DocTitle", parent=base["Title"], fontSize=19,
                           textColor=colors.HexColor("#275CB2"))
    return title, h1, h2, body, formula, callout


_TH = ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                     textColor=colors.white)
_TD = ParagraphStyle("TD", fontName="Helvetica", fontSize=8.5, leading=11)


def _table(rows, col_widths=None):
    # Cells are wrapped in Paragraphs so that long values (the genre vocabulary,
    # the exclusion criteria) wrap inside the column instead of running off the
    # page, which a bare string in a reportlab Table does.
    wrapped = [[Paragraph(str(cell), _TH) for cell in rows[0]]]
    wrapped += [[Paragraph(str(cell), _TD) for cell in row] for row in rows[1:]]

    table = Table(wrapped, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#275CB2")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fc")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_report_pdf() -> bytes:
    corpus = get_movie_corpus()
    df = corpus.df
    _, encoder = get_features()
    genres = genre_vocabulary(df)
    cfg = DEFAULT_CONFIG

    # The elicitation budget: both conditions show the same number of films.
    movies_pairwise = cfg.n_pairwise_trials * 2
    movies_ranking = cfg.n_ranking_trials * cfg.ranking_size
    relations_pairwise = cfg.n_pairwise_trials
    relations_ranking = cfg.n_ranking_trials * (
        cfg.ranking_size * (cfg.ranking_size - 1) // 2)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Project 4 — Preference Elicitation",
        author="Human-Centric Artificial Intelligence",
    )
    title, h1, h2, body, formula, callout = _styles()
    S = []

    def para(text, style=body):
        S.append(Paragraph(text, style))

    def bullets(items):
        S.append(ListFlowable(
            [ListItem(Paragraph(text, body), leftIndent=14) for text in items],
            bulletType="bullet", start="•", leftIndent=14,
        ))
        S.append(Spacer(1, 4))

    # ── Title ────────────────────────────────────────────────────────────────
    para("Project 4: Preference Elicitation", title)
    para(
        "Human-Centric Artificial Intelligence &mdash; a user study comparing "
        "pairwise choice against ten-film ranking for learning a new user's taste, "
        "on the IMDB 5000 Movie Dataset.", body)
    para(
        "<i>This report describes a study design and a working implementation of "
        "the participant interface. The study has not been conducted, so no "
        "participant results are reported anywhere in this document.</i>", body)
    S.append(Spacer(1, 6))

    # ── 1. Introduction ─────────────────────────────────────────────────────
    para("1. Introduction", h1)
    para(
        "A recommender system meeting a user for the first time knows nothing about "
        "them. It cannot use collaborative filtering, because that needs a rating "
        "history the user does not have. This is the <b>cold-start</b> problem, and "
        "the usual remedy is <b>preference elicitation</b>: ask the user a small "
        "number of deliberately chosen questions and infer their taste from the "
        "answers.")
    para(
        "How you ask matters. A choice between two films is quick and almost "
        "effortless, but each answer carries a single bit of preference "
        "information. Ranking ten films yields far more relations at once, but "
        "takes longer and asks more of the user's attention and working memory. "
        "Which of these buys more knowledge of the user for the effort it costs is "
        "an empirical question about people, not a question that can be settled by "
        "analysis of the model alone &mdash; which is exactly why it needs a user "
        "study.")
    para(
        "This project takes both interfaces down to the same underlying model, so "
        "that any difference measured between them is attributable to the "
        "<i>interaction</i> rather than to the mathematics. Every film is a feature "
        f"vector <b>x</b> in R^{encoder.dim}, every participant is a latent "
        "preference vector <b>w</b>, and utility is linear:")
    para("U(x)  =  w^T x", formula)
    para(
        "Both interfaces estimate the same <b>w</b> in the same space, and both are "
        "judged on the same held-out questions.")

    # ── 2. Dataset ──────────────────────────────────────────────────────────
    para("2. Dataset", h1)
    para(
        "We use the <b>IMDB 5000 Movie Dataset</b> (<font face='Courier'>"
        "movie_metadata.csv</font>), which contains metadata for roughly five "
        "thousand films: title, release year, duration, genres, content rating, "
        "country, language, cast and crew names, and popularity figures such as "
        "IMDB score, gross revenue and Facebook likes.")
    para(
        "<b>The dataset contains no user ratings and no preference labels at all.</b> "
        "That absence is the point of the project rather than an obstacle to work "
        "around: there is no ground-truth taste to look up, so every preference "
        "observation has to come from the participant in front of us.")

    para("2.1 Cleaning and missing values", h2)
    para(
        "A film is only usable if a participant can judge it and the encoder can "
        "represent it, so we drop any row missing <b>genres</b>, <b>duration</b> or "
        "<b>release year</b> &mdash; the three fields that carry the most weight in a "
        "quick judgement and that have no sensible imputed value. Missing "
        "<b>content rating</b> and <b>country</b> are kept and recorded as "
        "<i>Other</i>, because absence there is informative and dropping the rows "
        "would discard otherwise perfectly usable films. Titles arrive with a "
        "trailing non-breaking space in this dataset and are stripped; duplicate "
        "(title, year) pairs are removed so that the same film cannot appear twice "
        "in one ranking task.")
    S.append(_table([
        ["Property", "Value"],
        ["Films after cleaning", f"{corpus.n_movies:,}"],
        ["Distinct genres", str(len(genres))],
        ["Release years", f"{int(df['title_year'].min())}–{int(df['title_year'].max())}"],
        ["Missing values in the retained columns", "none"],
        ["User ratings available", "none — preferences are elicited"],
    ], col_widths=[8 * cm, 8 * cm]))
    S.append(Spacer(1, 6))

    # ── 3. Task 1 — Feature representation ──────────────────────────────────
    para("3. Task 1 &mdash; Feature representation", h1)
    para(
        "The representation has to serve an unusual constraint: <b>w</b> is "
        "estimated from roughly a dozen interactions, so the dimension must stay "
        "small enough to be identifiable from very little data, while remaining "
        "meaningful enough that the weights say something recognisable about taste. "
        f"We use <b>d = {encoder.dim}</b> features in four groups.")
    rows = [["Group", "d", "What it encodes"]]
    for group in describe_feature_groups(encoder):
        rows.append([group["name"], str(group["dims"]), group["detail"]])
    S.append(_table(rows, col_widths=[4.6 * cm, 1.2 * cm, 10.2 * cm]))
    S.append(Spacer(1, 6))

    para("3.1 What is included, and why", h2)
    bullets([
        f"<b>Genres ({len(genres)} multi-hot features).</b> The genre field is "
        "multi-valued, so a film contributes a 1 to every genre it lists. Genres "
        "are the closest thing the dataset has to a direct description of content, "
        "and they read naturally as preference dimensions: a weight on "
        "<i>Horror</i> is interpretable in a way that a weight on a latent factor "
        "is not.",
        f"<b>Numeric ({len(NUMERIC_COLUMNS)} features: duration, release year).</b> "
        "Both are standardized to zero mean and unit variance over the corpus, so "
        "that a coefficient is read per standard deviation and neither dominates "
        "the optimisation through its raw scale.",
        f"<b>Content rating ({len(RATING_BUCKETS)} one-hot features).</b> The raw "
        "field has many rare and historical values (<i>Approved</i>, <i>GP</i>, "
        "<i>M</i>, <i>X</i>, various TV codes). Fitting a weight per value would "
        "spend dimensions on categories a participant may never see, so we collapse "
        f"to {', '.join(RATING_BUCKETS)} by audience rather than by issuing body.",
        "<b>Country (1 binary feature: made in the USA).</b> The country field is "
        "dominated by the USA; a full one-hot encoding would add dozens of columns "
        "that are almost always zero. The single indicator keeps the one "
        "distinction that carries most of the signal.",
    ])

    para("3.2 What is deliberately excluded", h2)
    bullets([
        "<b>IMDB score, gross revenue, vote counts, Facebook likes.</b> These "
        "measure how famous or well-received a film is in general, not whether "
        "<i>this</i> participant would enjoy it. Including them would let the model "
        "explain choices by popularity and score well while learning nothing about "
        "personal taste &mdash; the model would become a popularity predictor "
        "wearing a preference model's clothes.",
        "<b>Language.</b> Over ninety per cent of the corpus is English, so the "
        "feature is nearly constant and its weight would be estimated from a "
        "handful of films.",
        "<b>Actor, director and plot-keyword identities.</b> These are genuinely "
        "predictive of taste but have thousands of levels. With a dozen "
        "observations per participant, a weight per name is unidentifiable. They "
        "are shown to the participant on screen (they help a person judge a film) "
        "but are not given to the model.",
    ])
    para(
        "Binary features are left at 0/1 rather than standardized. Standardizing "
        "them would make a weight mean &lsquo;per standard deviation of a rare "
        "genre&rsquo;, which inflates weights on rare genres and makes the fitted "
        "vector much harder to read; at 0/1 a weight is simply the change in "
        "utility from a film having that property.")

    para("3.3 Implementation", h2)
    para(
        "<font face='Courier'>fit_movie_encoder(df)</font> learns the genre "
        "vocabulary and the numeric means and standard deviations; "
        "<font face='Courier'>transform_movies(df, encoder)</font> applies them and "
        f"returns X with shape ({corpus.n_movies:,}, {encoder.dim}). Fitting is "
        "separated from transforming so that the encoding is reproducible and "
        "identical for both conditions &mdash; using different representations "
        "would confound the comparison the whole study is built to make.")

    # ── 4. Pairwise preference model ────────────────────────────────────────
    para("4. Pairwise preference model", h1)
    para(
        "Each film has a latent score s_i = w^T x_i. The Bradley&ndash;Terry model "
        "turns a pair of scores into a choice probability:")
    para(
        "P(i &gt; j | w)  =  exp(s_i) / [exp(s_i) + exp(s_j)]<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;=  sigmoid( w^T (x_i - x_j) )", formula)
    para(
        "The further apart the two scores, the more reliably the participant picks "
        "the higher one; equal scores give a coin flip. Only the <i>difference</i> "
        "of feature vectors enters, so anything a pair has in common cancels.")
    para("For observed choices (chosen_t, rejected_t):")
    para(
        "L_pair(w)  =  sum_t  log sigmoid( w^T (x_chosen,t - x_rejected,t) )",
        formula)
    para(
        f"With d = {encoder.dim} parameters and only "
        f"{cfg.n_pairwise_trials} observations the maximum-likelihood problem is "
        "under-determined, and perfectly separable answers would send ||w|| to "
        "infinity. We therefore maximise a regularized objective")
    para(
        f"max_w   L(w) - lambda * ||w||_2^2 &nbsp;&nbsp;&nbsp;"
        f"( lambda = {DEFAULT_REGULARIZATION} )", formula)
    para(
        "which is MAP estimation under a zero-mean Gaussian prior. This is not "
        "cosmetic: it prevents overconfidence from few comparisons and keeps a participant who "
        "answers consistently from producing an arbitrarily confident model.")

    # ── 5. Task 2 — Ranking model ───────────────────────────────────────────
    para("5. Task 2 &mdash; Extending Bradley&ndash;Terry to full rankings", h1)
    para(
        "A ranking is read as <b>repeated choice from what remains</b>: name the "
        "favourite of all ten, remove it, name the favourite of the remaining nine, "
        "and so on. Applying the Luce choice rule at each step gives the "
        "Plackett&ndash;Luce likelihood of a complete ordering:")
    para(
        "P(i_1 &gt; i_2 &gt; ... &gt; i_n | w)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;=  prod_{k=1..n-1}  "
        "exp(s_i_k) / sum_{j&gt;=k} exp(s_i_j)", formula)
    para(
        "The product stops at n-1 because once nine films have been placed the "
        "tenth is determined, contributing a factor of exactly 1. In log form:")
    para(
        "log P(ranking | w)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;=  sum_{k=1..n-1} "
        "[ s_i_k - logsumexp(s_i_k, ..., s_i_n) ]", formula)

    para("5.1 Why this extension", h2)
    bullets([
        "<b>At n = 2 it is exactly Bradley&ndash;Terry.</b> The single factor "
        "becomes exp(s<sub>1</sub>)/[exp(s<sub>1</sub>)+exp(s<sub>2</sub>)]. This "
        "is verified numerically in the test suite, where the difference between "
        "the two formulations is 0 to machine precision. It matters for the study: "
        "the two conditions are not merely analogous models, one is a strict "
        "generalisation of the other.",
        "<b>It reuses the same score w<sup>T</sup>x</b>, so neither interface gets "
        "an advantage from a richer parameterisation.",
        "<b>It is a proper likelihood over orderings</b> &mdash; the probabilities "
        "of all n! permutations sum to 1 (also unit-tested, for all 24 permutations "
        "of four items) &mdash; so w is directly estimable by the same MAP "
        "procedure, with the same lambda and the same optimiser.",
    ])

    para("5.2 The alternative we rejected", h2)
    para(
        f"A ranking of ten films could instead be expanded into its "
        f"{cfg.ranking_size * (cfg.ranking_size - 1) // 2} implied pairwise "
        "comparisons and fed to ordinary Bradley&ndash;Terry. That is simpler, but "
        "it is wrong about the data: those relations come from a <i>single</i> "
        "elicitation act and are strongly dependent, so treating them as "
        "independent observations multiplies the apparent evidence roughly forty-"
        "fivefold and produces a badly overconfident model. Since the study exists "
        "to compare how much each interface really tells us, an error that inflates "
        "one condition's evidence would corrupt the primary result. The sequential "
        "formulation is the honest direct likelihood for an ordering.")

    para("5.3 Estimation, identical for both conditions", h2)
    para(
        "Both objectives are minimised with L-BFGS-B from w = 0 using analytic "
        "gradients, with the same regularization. Holding the optimiser, the "
        "starting point, the prior and the feature space fixed across conditions "
        "means the only thing that differs between the two fitted models is the "
        "data the participant produced.")

    # ── 6. Task 3 — Research question and hypotheses ────────────────────────
    para("6. Task 3 &mdash; Research question and hypotheses", h1)
    para(
        "The study asks:")
    para(
        "<b>Which elicitation interface learns a new user&rsquo;s film preferences "
        "more effectively for the same number of films inspected: repeated pairwise "
        "choices, or rankings of ten?</b>", callout)
    S.append(_table([
        ["", "Statement"],
        ["H1 (primary)",
         "For an equal number of films inspected, the ranking interface yields a "
         "preference model with lower held-out log loss than the pairwise "
         "interface."],
        ["H0 (primary)",
         "The mean within-participant difference in held-out log loss between the "
         "two interfaces is zero."],
        ["H2 (secondary)",
         "Ranking ten films takes longer per film inspected and is rated as "
         "requiring more mental effort than pairwise choice."],
    ], col_widths=[3.2 * cm, 12.8 * cm]))
    S.append(Spacer(1, 6))
    para(
        "H1 and H2 are deliberately in tension. If ranking wins on accuracy while "
        "costing more effort, the interesting result is the exchange rate between "
        "the two, not either measure alone.")

    # ── 7. Experimental design ──────────────────────────────────────────────
    para("7. Experimental design", h1)
    para(
        "<b>Independent variable:</b> elicitation method in {pairwise, "
        "ranking}, manipulated <b>within subjects</b>. Tastes in film vary "
        "enormously between people, and that variance would swamp the interface "
        "effect in a between-subjects design; comparing each participant against "
        "themselves removes it entirely.")

    para("7.1 Counterbalancing", h2)
    para(
        "A within-subject design introduces order, learning and fatigue effects, so "
        "condition order is counterbalanced: group <b>AB</b> does pairwise then "
        "ranking, group <b>BA</b> the reverse. Participants are assigned "
        "alternately as sessions are created, which keeps the two groups balanced "
        "at every point during data collection rather than only in expectation.")

    para("7.2 The elicitation budget", h2)
    para(
        "Comparing an equal <i>number of tasks</i> would be meaningless, since one "
        "ranking of ten contains far more preference information than one pairwise "
        "choice. We considered three possible comparison budgets &mdash; equal "
        "time, equal films inspected, equal preference relations &mdash; we make "
        "<b>equal number of distinct films inspected</b> the primary budget:")
    S.append(_table([
        ["", "Pairwise", "Ranking"],
        ["Tasks", str(cfg.n_pairwise_trials), str(cfg.n_ranking_trials)],
        ["Films inspected", str(movies_pairwise), str(movies_ranking)],
        ["Preference relations obtained", str(relations_pairwise),
         f"{relations_ranking} (dependent)"],
    ], col_widths=[7 * cm, 4.5 * cm, 4.5 * cm]))
    S.append(Spacer(1, 6))
    para(
        "Both conditions therefore expose the participant to exactly "
        f"<b>{movies_pairwise} films</b>. This budget is chosen because the film is "
        "the unit the participant actually has to read and evaluate: it is the "
        "closest available proxy for the cognitive work being demanded, it is "
        "exactly equalisable in advance (unlike time, which cannot be fixed without "
        "rushing people), and it makes the comparison a fair test of the question "
        "the recommender designer really faces &mdash; <i>given that I may show a "
        "new user thirty films, how should I ask about them?</i> Elapsed time is "
        "measured throughout and analysed as the secondary efficiency outcome, so "
        "nothing is lost by not making it the budget.")
    para(
        f"The table also shows the tension the study probes: the same "
        f"{movies_pairwise} films yield {relations_pairwise} independent relations "
        f"under pairwise choice, but {relations_ranking} strongly dependent ones "
        "under ranking. Whether that extra structure survives the added cognitive "
        "load is exactly what H1 tests.")

    para("7.3 Movie sampling and carry-over control", h2)
    para(
        "Uniform random sampling is used; the design work is in the <i>allocation</i>. "
        "Each session draws one pool "
        f"of {cfg.movies_needed} distinct films from a seed stored with the "
        "session, and splits it into four <b>disjoint</b> parts: practice, "
        "pairwise, ranking, and held-out evaluation. No participant ever sees the "
        "same film twice anywhere in the study.")
    bullets([
        "<b>Disjoint condition sets</b> remove memory and recognition carry-over: a "
        "participant cannot be helped in the second condition by having thought "
        "about a film in the first.",
        "<b>A disjoint held-out set</b> means the final evaluation measures "
        "generalisation to unseen films rather than recall of ones already judged.",
        "<b>A stored seed</b> regenerates the entire trial plan deterministically, "
        "so the database holds one integer instead of a duplicated trial list and "
        "any session can be reconstructed exactly for audit.",
    ])

    para("7.4 Shared held-out evaluation", h2)
    para(
        f"After both conditions, every participant answers {cfg.n_heldout_pairs} "
        "further pairwise questions on films used nowhere else. Both fitted models "
        "predict those same questions. The evaluation is in pairwise form for both "
        "conditions because both models induce a probability over any pair, which "
        "makes one common yardstick possible; the alternative &mdash; testing each "
        "condition in its own format &mdash; would confound interface with test. "
        "These responses are used only for evaluation and never for refitting.")

    # ── 8. Participants and recruitment ─────────────────────────────────────
    para("8. Participants and recruitment", h1)
    S.append(_table([
        ["Item", "Plan"],
        ["Target population",
         "Adults (18+) who watch films at least occasionally, can read the "
         "interface language, and can operate a standard web browser."],
        ["Inclusion criteria",
         "18 or over; self-reported film watching at least monthly; consent given."],
        ["Exclusion criteria (pre-registered)",
         "Did not complete both conditions; median pairwise response time under "
         "800 ms (implausibly fast for reading two films); all rankings submitted "
         "unchanged in under five seconds; technical failure losing interaction "
         "data."],
        ["Recruitment channel",
         "University participant pool and student mailing lists, with a public "
         "link for snowball recruitment. Fully remote and self-service."],
        ["Compensation",
         "Course credit where the participant pool allows it, otherwise entry into "
         "a small prize draw. Compensation is not contingent on completing the "
         "study, so withdrawal carries no penalty."],
        ["Context",
         "Online, unsupervised, participant's own device, desktop or laptop "
         "recommended for the ranking interface."],
    ], col_widths=[4.5 * cm, 11.5 * cm]))
    S.append(Spacer(1, 6))

    para("8.1 Sample-size planning", h2)
    para(
        "The specification does not prescribe a participant count, so we plan one "
        "rather than assert one. The primary comparison is a paired test of a "
        "within-participant difference, for which the required sample size at "
        f"significance alpha = {ALPHA} (two-sided) and power {POWER:.0%} is "
        "approximately")
    para(
        "n  ~=  (z_{1-alpha/2} + z_{1-beta})^2 / d^2<br/>"
        f"&nbsp;&nbsp;&nbsp;~=  (1.96 + 0.84)^2 / {TARGET_EFFECT}^2  ~=  32",
        formula)
    para(
        f"for a medium standardized effect d = {TARGET_EFFECT}. Correcting the "
        f"normal approximation for use of the t distribution gives roughly "
        f"<b>n = {PLANNED_N}</b> completed participants. We would recruit "
        f"<b>{RECRUIT_N}</b> to absorb the pre-registered exclusions and dropout, "
        "and would stop at a fixed target rather than checking significance as data "
        "arrives, since repeatedly testing an accumulating sample inflates the "
        "false-positive rate well above the nominal 5%. A medium effect is the "
        "smallest difference we would consider practically interesting for an "
        "interface choice; the study is not powered to detect smaller ones, and we "
        "would say so rather than treat a null result as proof of equivalence.")

    # ── 9. Procedure ────────────────────────────────────────────────────────
    para("9. Procedure", h1)
    para(
        "The implemented interface walks the participant through twelve steps. The "
        "server holds the current step and redirects any request to the page the "
        "participant is genuinely on, so the sequence cannot be skipped, replayed "
        "or reordered by editing a URL, and a reload never double-counts an answer.")
    S.append(_table([
        ["#", "Step", "What happens"],
        ["1", "Information &amp; consent",
         "Purpose, what is stored, withdrawal rights, explicit opt-in."],
        ["2", "Background questionnaire",
         "Age range, film-watching frequency, familiarity with recommenders. "
         "Nothing identifying."],
        ["3", "Instructions",
         "Both interfaces explained; participants told to answer for themselves, "
         "not for what is objectively the better film."],
        ["4", "Practice",
         "One unscored task in <i>each</i> interface, in the order the participant "
         "will meet them, on films used nowhere else. Stored separately and "
         "excluded from analysis."],
        ["5", "First condition",
         f"{cfg.n_pairwise_trials} pairwise choices or {cfg.n_ranking_trials} "
         "rankings, per counterbalancing group."],
        ["6", "Condition questionnaire",
         "Ease, mental effort, expressiveness, confidence, willingness to continue; "
         "seven-point scales plus optional free text."],
        ["7", "Break", "Self-paced pause; previews the interface that comes next."],
        ["8", "Second condition", "The other interface, on a disjoint film set."],
        ["9", "Condition questionnaire", "The same items, about the second interface."],
        ["10", "Held-out check",
         f"{cfg.n_heldout_pairs} pairwise questions on unseen films; identical for "
         "both conditions."],
        ["11", "Final comparison",
         "Direct head-to-head preference between the interfaces, asked "
         "<i>before</i> any results are shown."],
        ["12", "Debrief",
         "Purpose explained, both fitted models shown to the participant, "
         "withdrawal route restated."],
    ], col_widths=[0.9 * cm, 3.6 * cm, 11.5 * cm]))
    S.append(Spacer(1, 6))
    para(
        "Two ordering decisions are load-bearing. Practice covers <b>both</b> "
        "interfaces before any measurement begins, so neither condition benefits "
        "from a warm-up the other did not get. And the final comparison is asked "
        "<b>before</b> the debrief reveals which model predicted better, because "
        "showing a participant that result first would contaminate the very "
        "judgement being collected.")

    para("9.1 Pilot", h2)
    para(
        "The protocol would first be piloted with 5&ndash;10 participants to check "
        "instruction clarity, task duration, and that the "
        "logging captures what the analysis needs. <b>Pilot data would not be "
        "merged into the main dataset</b>, and any change to the protocol after "
        "piloting would be frozen before real collection begins.")

    # ── 10. Measures ────────────────────────────────────────────────────────
    para("10. Measures", h1)
    S.append(_table([
        ["Kind", "Measure", "How obtained"],
        ["Objective (primary)", "Held-out log loss",
         "Mean -log P(observed choice) of the fitted w over the "
         f"{cfg.n_heldout_pairs} shared held-out pairs."],
        ["Objective", "Held-out accuracy",
         "Fraction of held-out pairs where the model's preferred film was the one "
         "chosen."],
        ["Interaction", "Time per task, time per film inspected",
         "Response time recorded client-side per task, range-validated on the "
         "server."],
        ["Interaction", "Tasks completed, films inspected",
         "Derived from the stored trial log."],
        ["Subjective", "Ease, mental effort, expressiveness, confidence, willingness",
         "Seven-point scales after each condition."],
        ["Subjective", "Interface preference",
         "Forced choice between the two interfaces on overall preference, "
         "expressiveness and effort."],
    ], col_widths=[3.2 * cm, 5 * cm, 7.8 * cm]))
    S.append(Spacer(1, 6))
    para(
        "<b>Log loss is the primary outcome rather than accuracy.</b> Accuracy over "
        f"{cfg.n_heldout_pairs} pairs can take only {cfg.n_heldout_pairs + 1} "
        "distinct values, so it discards most of the information in the model's "
        "predictions and ties between conditions are common. Log loss uses the "
        "predicted probability itself, rewarding a model that is confident when "
        "right and cautious when wrong, and gives a continuous per-participant "
        "score far better suited to a paired test. Accuracy is retained as a "
        "secondary, more interpretable outcome.")
    para(
        "Note also what is <i>not</i> measured: ||w-hat - w||. The "
        "participant's true preference vector does not exist as an observable "
        "quantity, so estimation error against it is not available and any figure "
        "claiming to be it would be fictional. Predictive performance on fresh "
        "judgements from the same person is the honest substitute.")

    # ── 11. Planned analysis ────────────────────────────────────────────────
    para("11. Planned analysis", h1)
    para("Fixed in advance, before any data is examined:")
    bullets([
        "<b>Cleaning.</b> Apply the pre-registered exclusions from section 8. "
        "Practice trials are excluded by construction, being stored in their own "
        "block. Participants are never removed for producing results that "
        "disagree with the hypothesis.",
        "<b>Descriptives.</b> Per-condition means and standard deviations of log "
        "loss, accuracy and time; per-participant paired differences plotted "
        "individually so that a mean difference cannot hide two opposing "
        "subgroups.",
        "<b>Primary test.</b> A paired comparison of held-out log loss, pairwise "
        "versus ranking, within participant. Normality of the differences is "
        "checked with Shapiro&ndash;Wilk; a paired t-test if it holds, a Wilcoxon "
        "signed-rank test otherwise. Two-sided, "
        f"alpha = {ALPHA}. Reported with the effect size and a confidence "
        "interval on the mean difference, not a p-value alone.",
        "<b>Secondary tests.</b> The same paired structure for accuracy, time per "
        "film and each subjective scale. This is a family of related tests, so "
        "p-values within it are corrected by the Holm&ndash;Bonferroni procedure; "
        "the single primary test is not corrected and is not reinterpreted in the "
        "light of the secondary ones.",
        "<b>Order effects.</b> Condition order is included as a factor to confirm "
        "counterbalancing worked; a significant order effect would be reported "
        "rather than quietly absorbed.",
    ])

    # ── 12. Ethics and data protection ──────────────────────────────────────
    para("12. Ethics and data protection", h1)
    bullets([
        "<b>Informed consent.</b> Explicit opt-in on the first screen, after a plain-"
        "language description of purpose, duration, what is stored and who to "
        "contact. The study cannot proceed without it.",
        "<b>Withdrawal without penalty.</b> A participant may close the browser at "
        "any point; incomplete sessions are excluded rather than analysed. "
        "Compensation is not contingent on completion.",
        "<b>Data minimisation.</b> No name, email, IP or any direct identifier is "
        "collected anywhere. The background questionnaire asks only for an age "
        "<i>range</i>, film-watching frequency and recommender familiarity &mdash; "
        "the three variables the analysis would actually use.",
        "<b>Pseudonymisation.</b> Each session is identified by a random 12-"
        "character code with no link to a person. The code is shown to the "
        "participant at the end, because it is the only way for them to request "
        "deletion &mdash; and precisely because we cannot otherwise find their "
        "record, which is the point.",
        "<b>GDPR.</b> Interaction sequences can be indirect identifiers even without "
        "names, so the logs are treated as personal data: "
        "lawful basis is consent, processing is limited to the stated research "
        "purpose, and participants have access and erasure rights via their code.",
        "<b>Storage and retention.</b> Data stored server-side, not in the "
        "browser. Raw interaction logs deleted after analysis and publication; "
        "only aggregate results retained. Export endpoints would require "
        "researcher authentication in a real deployment.",
        "<b>Ethics approval.</b> Institutional ethics review would be obtained "
        "before any recruitment, and exclusion criteria fixed before collection "
        "begins.",
        "<b>Debriefing.</b> The final page explains the purpose in full and shows "
        "the participant what was learned about them, so that taking part returns "
        "something to them rather than only to us.",
    ])

    # ── 13. Interface implementation ────────────────────────────────────────
    para("13. Interface implementation", h1)
    para(
        "The interface is a Django app (<font face='Courier'>project4</font>) in "
        "the shared repository, with the mathematics in a services layer "
        "that has no Django dependency and is unit-tested independently of the web "
        "layer.")

    para("13.1 Landing page", h2)
    para(
        "Project title, a description of the design, the dataset summary, example "
        "films, a <b>Download report</b> action serving this document, and a "
        "<b>Start the study</b> action creating a fresh pseudonymous session.")

    para("13.2 Pairwise interface", h2)
    bullets([
        "Exactly two films per trial, drawn from the session's pairwise pool, never "
        "duplicated within a pair.",
        "Each film card <i>is</i> the submit control, so nothing is preselected and "
        "a choice requires an explicit click. The server rejects any film id that "
        "was not one of the two displayed.",
        "Title, year, genres, duration, content rating, country, director and lead "
        "actor are shown &mdash; enough for a real judgement.",
        "Progress shown as <i>Task n of N</i>; response time recorded client-side "
        "and range-validated server-side, since a client value can never be trusted.",
        "No model output or recommendation is ever shown during elicitation, which "
        "would otherwise feed back into later answers and confound the result.",
    ])

    para("13.3 Ranking interface", h2)
    bullets([
        f"Exactly {cfg.ranking_size} films per task, in an ordered list with rank 1 "
        "at the top explicitly labelled <i>most preferred</i>.",
        "Reordering by drag-and-drop <b>or</b> by per-row up/down buttons. The "
        "buttons are real focusable controls, so the task is completable by "
        "keyboard and with a screen reader; an ARIA live region announces each "
        "move.",
        "The buttons are submit controls that JavaScript intercepts, so with "
        "scripting disabled each press posts and the server performs the identical "
        "swap. The interface degrades rather than breaking.",
        "The whole order is posted back and accepted only if it is exactly a "
        "permutation of the films sent &mdash; one check that rejects duplicates, "
        "omissions, extras and unknown ids. A tampered order is discarded rather "
        "than stored partially correct.",
    ])

    para("13.4 Study state and logging", h2)
    para(
        "Four models persist the study: <font face='Courier'>StudySession</font> "
        "(participant code, condition order, seed, current step, timestamps), "
        "<font face='Courier'>PairwiseTrial</font> and "
        "<font face='Courier'>RankingTrial</font> (both tagged with a block "
        "&mdash; practice, main or held-out &mdash; the films shown, the response "
        "and the response time), and "
        "<font face='Courier'>QuestionnaireResponse</font>. A fifth, "
        "<font face='Courier'>PreferenceModelFit</font>, stores each condition's "
        "fitted w with its held-out scores.")
    para(
        "The current trial index is derived from how many trials of that block are "
        "already stored rather than held in a separate cursor. That single choice "
        "makes the flow refresh-safe and resumable without any reconciliation "
        "logic: there is no second copy of the position to fall out of step with "
        "the data. Anonymised logs can be exported as CSV at session and trial "
        "level for analysis.")

    # ── 14. Limitations and extensions ──────────────────────────────────────
    para("14. Limitations and extensions", h1)
    para("<b>Limitations of the design.</b>")
    bullets([
        "<b>Linear utility.</b> U(x) = w<sup>T</sup>x cannot express interactions "
        "such as enjoying comedy horror while disliking both alone. Linearity buys "
        "identifiability from very few observations, which is the binding "
        "constraint here, but it is a real ceiling on what can be learned.",
        "<b>Stated versus revealed preference.</b> What a participant clicks in a "
        "study is not necessarily what they would choose on a Friday evening. The "
        "held-out check inherits the same bias, so it validates internal "
        "consistency rather than real viewing behaviour.",
        "<b>Unfamiliar films add noise.</b> Judging an unrecognised film from "
        "metadata is a different, noisier task than choosing between two known "
        "ones, and random sampling from five thousand films makes that common.",
        "<b>Cognitive load of ranking ten.</b> Ordering ten items is genuinely "
        "demanding, and a fatigued participant may order the top few carefully and "
        "the rest arbitrarily &mdash; which the Plackett&ndash;Luce likelihood, "
        "weighting every position, would take at face value.",
        "<b>Random sampling is sample-inefficient.</b> Many random pairs are "
        "lopsided and nearly uninformative about w.",
        "<b>Single session, single sitting.</b> Preference stability over time is "
        "not measured.",
    ])
    para("<b>Extensions worth pursuing.</b>")
    bullets([
        "<b>Adaptive query selection</b>: choose pairs whose predicted "
        "probability is nearest 0.5, or ranking sets spanning still-uncertain "
        "directions of w. This attacks the sample-inefficiency limitation "
        "directly and would form a natural third condition.",
        "<b>Bayesian estimation of w</b> instead of MAP, giving calibrated "
        "uncertainty rather than a point estimate &mdash; and a principled way to "
        "know when enough has been asked, which is the real goal of elicitation.",
        "<b>Position-weighted ranking likelihood</b>, discounting lower positions "
        "to reflect that people rank their favourites more carefully than their "
        "least favourites.",
        "<b>Equal-time budget as a second study</b>, complementing the "
        "equal-films budget used here and letting the two budgets triangulate the "
        "efficiency question.",
    ])

    doc.build(S)
    return buf.getvalue()
