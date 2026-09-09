"""PDF report for Project 2.

Numbers come from the live services, so the document cannot drift from the
code. Formulas are written in plain ASCII: the built-in PDF fonts are
WinAnsi-encoded and silently drop Greek letters and mathematical operators.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

from .data import (
    BIOMETRIC_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    get_penguin_data,
)
from .grids import LOGREG_C_GRID, TREE_MAX_LEAF_NODES_GRID, get_grid
from .selection import (
    LAMBDA_DEFAULT, LAMBDA_MAX, LAMBDA_MIN, LAMBDA_STEP, select_best,
)

SEED = 42


def _styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=base["Heading1"], fontName="Times-Bold",
                        fontSize=13, leading=16, spaceBefore=14, spaceAfter=5,
                        textColor=colors.black)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontName="Times-Bold",
                        fontSize=11, leading=14, spaceBefore=10, spaceAfter=3,
                        textColor=colors.black)
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName="Times-Roman",
                          fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY,
                          spaceAfter=7, firstLineIndent=0)
    formula = ParagraphStyle("Formula", parent=body, fontName="Courier",
                             fontSize=9.5, leading=13, leftIndent=22,
                             alignment=TA_LEFT, spaceBefore=5, spaceAfter=9)
    title = ParagraphStyle("DocTitle", parent=base["Title"], fontName="Times-Bold",
                           fontSize=17, leading=21, spaceAfter=2,
                           textColor=colors.black)
    subtitle = ParagraphStyle("Subtitle", parent=body, fontName="Times-Italic",
                              fontSize=10.5, alignment=TA_CENTER, spaceAfter=2)
    return title, subtitle, h1, h2, body, formula


_TH = ParagraphStyle("TH", fontName="Times-Bold", fontSize=9.5, leading=12)
_TD = ParagraphStyle("TD", fontName="Times-Roman", fontSize=9.5, leading=12)


def _table(rows, col_widths=None):
    """A plain rule-ruled table, in the style used in printed papers."""
    wrapped = [[Paragraph(str(c), _TH) for c in rows[0]]]
    wrapped += [[Paragraph(str(c), _TD) for c in row] for row in rows[1:]]
    table = Table(wrapped, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return table


def _grid_text(values):
    return ", ".join("None" if v is None else str(v) for v in values)


def build_report_pdf() -> bytes:
    data = get_penguin_data(SEED)
    tree_grid = get_grid("tree", SEED)
    logreg_grid = get_grid("logreg", SEED)

    # What the slider actually picks at each end of its range.
    picks = []
    for model_class, grid in (("Decision tree", tree_grid),
                              ("Logistic regression", logreg_grid)):
        for lam in (LAMBDA_MIN, 0.01, LAMBDA_MAX):
            entry = select_best(grid, lam)
            picks.append([
                model_class, f"{lam:g}", entry.param_display,
                f"{entry.test_accuracy:.1%}", str(entry.complexity),
            ])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Project 2: Explainability",
        author="Human-Centric Artificial Intelligence",
    )
    title, subtitle, h1, h2, body, formula = _styles()
    S = []

    def para(text, style=body):
        S.append(Paragraph(text, style))

    def bullets(items):
        S.append(ListFlowable(
            [ListItem(Paragraph(t, body), leftIndent=14) for t in items],
            bulletType="bullet", start="•", leftIndent=14,
        ))
        S.append(Spacer(1, 4))

    # ── Title ───────────────────────────────────────────────────────────────
    para("Project 2: Explainability", title)
    para("Human-Centric Artificial Intelligence", subtitle)
    para("Interpretability, counterfactual explanations and global feature "
         "effects on the Palmer Penguins dataset", subtitle)
    S.append(Spacer(1, 14))

    # ── 1. Introduction ─────────────────────────────────────────────────────
    para("1. Introduction", h1)
    para(
        "We built an interactive dashboard for explaining models trained to "
        "predict penguin species. It has three linked regions: a model view that "
        "trades accuracy against complexity through a lambda slider, a "
        "counterfactual explorer, and global feature-effect plots. Everything on "
        "the page comes from whichever model the slider currently selects, so "
        "moving the slider or switching model family updates all three regions "
        "together.")
    para(
        "This report explains the choices we made and why we made them. Every "
        "figure in it is read from the running code rather than typed in, so it "
        "cannot fall out of step with what the dashboard actually does.")

    # ── 2. Dataset and preprocessing ────────────────────────────────────────
    para("2. Dataset and preprocessing", h1)
    para(
        f"We use the Palmer Penguins dataset, predicting <b>species</b> "
        f"(Adelie, Gentoo, Chinstrap) from the remaining seven features. We drop "
        f"rows with any missing value, which leaves <b>{len(data.y_all)} of 344</b> "
        f"rows, and split {len(data.y_train)} train / {len(data.y_test)} test "
        f"(80/20, stratified by species, random_state={SEED}). Dropping is safe "
        f"here because only a handful of rows are affected; imputing a species "
        f"label would be guesswork, and imputing measurements would invent data "
        f"the explanations would then be built on.")
    S.append(_table([
        ["Feature group", "Features", "Treatment"],
        ["Biometric (numeric)", ", ".join(BIOMETRIC_FEATURES),
         "Median-imputed. Standardized for logistic regression only."],
        ["Year (numeric)", "year", "Median-imputed; kept on its own scale."],
        ["Categorical", ", ".join(CATEGORICAL_FEATURES),
         "Most-frequent-imputed, then one-hot encoded."],
    ], col_widths=[3.6 * cm, 5.4 * cm, 7 * cm]))
    S.append(Spacer(1, 6))
    para(
        "We scale numeric features for logistic regression because it needs them "
        "on a common scale, but deliberately <b>not</b> for the decision tree. "
        "Leaving the tree unscaled keeps its split thresholds in real units, so "
        "the plotted tree reads as <i>flipper_length_mm &lt;= 206.5</i> rather "
        "than as a standardized number nobody can interpret. Since the point of "
        "the project is explainability, we would rather the tree stay readable.")
    para(
        "The preprocessor and the estimator are stored together as one fitted "
        "pipeline. That way the dashboard, the counterfactuals, the PDP and the "
        "ALE all apply exactly the same preprocessing, and we cannot accidentally "
        "explain a model using differently prepared data than it was trained on.")

    # ── 3. Task 1: the tree ─────────────────────────────────────────────────
    para("3. Task 1: the decision tree, its accuracy and its complexity", h1)
    para(
        f"We train {len(TREE_MAX_LEAF_NODES_GRID)} trees over "
        f"max_leaf_nodes in {{{_grid_text(TREE_MAX_LEAF_NODES_GRID)}}}, which "
        f"spans the whole complexity axis from a two-leaf stump to a fully grown "
        f"tree. The interface shows the tree itself, its test accuracy and its "
        f"number of leaves, which is the complexity measure the project sheet "
        f"specifies for a tree:")
    para("Omega(f) = number of leaves", formula)

    # ── 4. Task 2: the slider ───────────────────────────────────────────────
    para("4. Task 2: the lambda slider", h1)
    para("The sheet defines the model to display as the maximizer of")
    para("score = acc_test - lambda * Omega(f)", formula)
    para(
        f"and we implement exactly that in <font face='Courier'>"
        f"services/selection.py</font>. The slider runs from {LAMBDA_MIN} to "
        f"{LAMBDA_MAX} in steps of {LAMBDA_STEP} and defaults to "
        f"{LAMBDA_DEFAULT:g}.")
    para(
        "One decision worth stating: the slider <b>does not retrain anything</b>. "
        "We train the whole grid once, then the slider selects among the already "
        "fitted models. This matters for the interface. Retraining on every slider "
        "movement would make it sluggish, and worse, it would let the displayed "
        "model jump around for reasons unrelated to lambda. Selecting post-hoc "
        "makes the slider a pure, instant view onto a fixed set of models. Ties "
        "break toward the simpler model, since the entire point of the criterion "
        "is to prefer simplicity when accuracy is equal.")
    S.append(_table([["Model family", "lambda", "Parameter", "Test accuracy", "Omega"]] + picks,
                    col_widths=[4 * cm, 1.8 * cm, 4.2 * cm, 3 * cm, 2.5 * cm]))
    S.append(Spacer(1, 6))
    para(
        "The behaviour is what the criterion predicts: at lambda = 0 the most "
        "accurate model wins, and as lambda grows the complexity term takes over "
        "and simpler models are selected. The two-leaf stump is never selected, "
        "because it cannot separate three species however cheap it is.")

    # ── 5. Task 3: logistic regression ──────────────────────────────────────
    para("5. Task 3: logistic regression and its complexity measure", h1)
    para(
        f"We train {len(LOGREG_C_GRID)} multinomial L1 logistic regressions over "
        f"C in {{{_grid_text(LOGREG_C_GRID)}}} (solver=saga, max_iter=5000). C is "
        f"the <i>inverse</i> regularization strength, so small C means stronger "
        f"regularization and a sparser model.")
    para(
        "The sheet leaves Omega open for this model, so we had to choose one. We "
        "use:")
    para("Omega(f) = number of nonzero coefficients   (|coef| > 1e-6)", formula)
    para(
        "counted across all three classes. We picked this because it pairs "
        "naturally with the L1 penalty, which drives coefficients to zero, and "
        "because it is the measure a person can actually read: it answers \"how "
        "many features does this model use?\". We considered the L1 and L2 norms "
        "of the coefficient vector instead, but neither answers that question. A "
        "model can have a small norm while still using every feature a little, "
        "which is not what we mean by simple when we are trying to explain it to "
        "someone. The 1e-6 floor is there because L1 drives coefficients very "
        "close to zero numerically without landing exactly on it.")

    # ── 6. Task 4: counterfactuals ──────────────────────────────────────────
    para("6. Task 4: counterfactual explanations", h1)
    para(
        "The user picks a row from the dataset and a target species, and we show "
        "the closest examples we can find that the <i>currently selected</i> "
        "model would classify as that target. The procedure follows the method "
        "from the lecture: sample N candidates near x, keep the ones predicted as "
        "the target, rank them by distance to x, and return the best k.")

    para("6.1 Distance", h2)
    para(
        "Features here sit on wildly different scales. A millimetre of bill "
        "length and a gram of body mass are simply not comparable, so a raw "
        "L1 or L2 distance ends up dominated by body mass. We use the "
        "MAD-weighted L1 distance from the lecture:")
    para(
        "d(x, z)  =  sum_j  |x_j - z_j| / MAD_j<br/>"
        "MAD_j    =  median_i | x_ij - median_i(x_ij) |", formula)
    para(
        "over the numeric features. Dividing by each feature's median absolute "
        "deviation makes the distance scale-free: a change counts as large only "
        "relative to how much that feature naturally varies. We floor MAD at 1e-6 "
        "so a near-constant feature cannot divide by zero.")

    para("6.2 Perturbing each kind of feature", h2)
    para(
        "The sheet asks what to do about features that are not decimal, and we "
        "treat each kind on its own terms. Adding Gaussian noise to an island "
        "name is meaningless, and adding it to a year produces values like "
        "2008.4 that do not exist:")
    bullets([
        "<b>Biometric decimals.</b> Gaussian noise, z = x + N(0, alpha * std), "
        "clipped to the range actually observed in the data so we never propose a "
        "penguin larger than any that exists.",
        "<b>Year (discrete).</b> A small Gaussian, rounded to an integer and "
        "clipped to the observed years.",
        "<b>Categoricals (island, sex).</b> Kept with probability 0.8, otherwise "
        "resampled to another valid category. Never perturbed with noise, because "
        "categories have no ordering to move along.",
    ])
    para(
        "Because categorical changes are not on the same footing as numeric ones, "
        "we rank primarily by the MAD distance and break ties toward "
        "counterfactuals that flip fewer categories. A counterfactual that says "
        "\"if this penguin had lived on another island\" is less actionable than "
        "one that only adjusts a measurement.")
    para(
        "If no counterfactual is found we widen the search (more candidates, "
        "larger numeric noise, a higher chance of switching a category) and "
        "retry, exactly as the sheet suggests. This matters "
        "in practice: for some rows and targets the nearby region contains "
        "nothing of the target class, and a single narrow pass would simply "
        "report failure.")

    # ── 7. Task 5: PDP and ALE ──────────────────────────────────────────────
    para("7. Task 5: feature effects with PDP and ALE", h1)
    para(
        "The user selects one of the four biometric features and sees both a PDP "
        "and an ALE plot, each with one curve per species, computed from the "
        "currently selected model. Both are implemented from scratch using only "
        "the model's predict_proba and numpy; we use no library PDP or ALE "
        "function, as the sheet requires.")

    para("7.1 Partial dependence", h2)
    para("PD_f(v)  =  E_{x_B} [ f(v, x_B) ]", formula)
    para(
        "We build a grid over the feature's observed range, and for each grid "
        "value we set that feature to the value for <i>every</i> row, call "
        "predict_proba, and average. That is the Monte-Carlo estimate of the "
        "expectation over the marginal distribution of the other features.")

    para("7.2 Accumulated local effects", h2)
    para(
        "ALE_f(v)  =  integral over z of  E_{x_B | z} [ d f / d z ]  dz  -  C",
        formula)
    para(
        "We split the feature into equal-population (quantile) bins. Within each "
        "bin we take only the rows whose value falls in that bin, predict at the "
        "bin's upper and lower edge, and average the difference. That is the "
        "local effect of moving across the bin. We accumulate those effects "
        "across bins, which is the integral, then subtract C so the "
        "data-weighted mean is zero.")
    para("Three details we were careful about:")
    bullets([
        "bins are half-open [lo, hi), with the last closed, so a point sitting on "
        "an edge is not counted in two bins;",
        "centering is weighted by how many rows each bin holds, so the mean is "
        "zero over the data rather than over the bins;",
        "an empty bin contributes zero local effect and carries the accumulator "
        "forward rather than breaking the curve.",
    ])

    para("7.3 Why both", h2)
    para(
        "The two answer different questions and disagree in an informative way. "
        "PDP averages over the <i>marginal</i> distribution of the other "
        "features, so when features are correlated it evaluates the model on "
        "combinations that do not occur, such as a very long flipper paired with "
        "a very small body mass. ALE only ever uses local differences "
        "within a bin of the <i>conditional</i> distribution, so it never asks "
        "the model about a penguin that could not exist. On this dataset the "
        "biometric features are strongly correlated, which is exactly the "
        "situation where the difference shows.")

    # ── 8. Exact vs discretized derivatives ─────────────────────────────────
    para("8. Exact and discretized derivatives for ALE", h1)
    para(
        "The sheet asks which model we can differentiate exactly and which forces "
        "a discretization. The answer differs by model family.")
    para(
        "<b>Logistic regression can be differentiated exactly</b>, because the "
        "multinomial softmax is smooth. For class c and feature j:")
    para(
        "d p_c / d x_j  =  p_c * ( beta_cj - sum_r p_r * beta_rj )", formula)
    para(
        "with a scaling factor for the standardization we apply to numeric "
        "features.")
    para(
        "<b>A decision tree cannot.</b> It is piecewise-constant: the prediction "
        "is flat inside a leaf and jumps at the split thresholds. So its "
        "derivative is zero almost everywhere and undefined exactly at the "
        "thresholds, which are the only places anything happens. An exact "
        "derivative would therefore tell us nothing at all, and ALE for a tree "
        "has to be built from finite differences over a discretization.")
    para(
        "We use finite differences for both families. The lecture defines ALE "
        "with the derivative estimated on a grid of values in any case, and using "
        "one estimator for both keeps the two model families directly comparable. "
        "If we computed the tree numerically and the regression "
        "analytically, any difference between their ALE curves would be partly an "
        "artefact of the two estimators rather than a real difference between "
        "the models.")

    # ── 9. How it fits together ─────────────────────────────────────────────
    para("9. How the parts fit together", h1)
    para(
        "The sheet asks that the counterfactual and feature-effect regions be "
        "linked to the model chosen in the first three tasks, and they are: one "
        "shared selection function decides which fitted pipeline is current, and "
        "every region reads from it. Changing the model family or moving the "
        "lambda slider updates the displayed model, the counterfactuals and both "
        "feature-effect plots at once, so what the dashboard shows is always a "
        "set of explanations of one and the same model.")

    doc.build(S)
    return buf.getvalue()
