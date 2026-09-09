"""Generate the mandatory PDF report for Project 3.

Builds a multi-section report (reportlab) from the cached experiment results:
introduction, dataset, baseline classifier, simulated expert, learning-to-defer,
active learning, the optional human interface, discussion, and conclusion.
Returned as raw PDF bytes for a download response.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from .active import get_active
from .baseline import MODEL_DESCRIPTION, get_baseline_eval
from .data import get_agnews
from .defer import get_deferral
from .expert import get_expert_eval


def _styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=15, spaceAfter=6,
                        textColor=colors.HexColor("#275CB2"))
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=12, spaceBefore=10,
                        spaceAfter=4, textColor=colors.HexColor("#1e4a9a"))
    body = ParagraphStyle("Body", parent=base["BodyText"], fontSize=10, leading=14,
                          alignment=TA_LEFT, spaceAfter=6)
    title = ParagraphStyle("Title", parent=base["Title"], fontSize=20,
                           textColor=colors.HexColor("#275CB2"))
    return title, h1, h2, body


def _pct(x):
    return f"{x * 100:.2f}%"


def _metric_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#275CB2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_report_pdf() -> bytes:
    data = get_agnews()
    base_ev = get_baseline_eval()
    exp_ev = get_expert_eval()
    dfr = get_deferral()
    act = get_active()
    adv, conf = dfr["advantage"], dfr["confidence"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Project 3 — Active Learning for Learning-to-Defer",
    )
    title, h1, h2, body = _styles()
    S = []

    def para(text, style=body):
        S.append(Paragraph(text, style))

    # ── Title ──
    para("Project 3: Active Learning for Learning-to-Defer", title)
    para("Human-Centric Artificial Intelligence &mdash; AG News topic classification", body)
    S.append(Spacer(1, 8))

    # 1. Introduction
    para("1. Introduction", h1)
    para(
        "This project builds a human&ndash;AI team for topic classification of news "
        "articles. A classifier predicts the topic, but for each article the system "
        "may instead <b>defer</b> the decision to a simulated human expert. Because "
        "expert labels are expensive, we use <b>active learning</b> to decide which "
        "articles are worth querying the expert on, so the team learns when deferral "
        "helps with as few queries as possible."
    )

    # 2. Dataset
    para("2. Dataset", h1)
    para(
        f"We use the AG News dataset (source: fancyzhx/ag_news), a topic "
        f"classification task with four classes: {', '.join(data.class_names)}. "
        f"The training split ({data.n_train} articles) is used for training; the "
        f"test split ({data.n_test} articles) is used only for evaluation, never for "
        f"model selection or active-learning query selection."
    )

    # 3. Baseline classifier
    para("3. Baseline Classifier", h1)
    para(
        f"The baseline is a {MODEL_DESCRIPTION}. Logistic regression is chosen over a "
        f"linear SVM because it exposes calibrated class probabilities, which both the "
        f"deferral rule and the active-learning uncertainty score rely on. It reaches "
        f"a test accuracy of <b>{_pct(base_ev['accuracy'])}</b>."
    )
    rows = [["Class", "Test accuracy"]]
    for i, c in enumerate(base_ev["class_names"]):
        rows.append([c, _pct(base_ev["per_class_accuracy"][i])])
    S.append(_metric_table(rows, col_widths=[5 * cm, 4 * cm]))
    para(
        "The classifier is strong on World and Sports but weaker on Business and "
        "Sci/Tech, which it confuses with each other &mdash; the region where an "
        "expert could help.", body)

    # 4. Simulated expert
    para("4. Simulated Expert", h1)
    para(
        f"{exp_ev['description']} Its competence region "
        f"({', '.join(exp_ev['competence_topics'])}) is chosen to complement the "
        f"classifier's weaknesses. The expert is imperfect and region-specific, with "
        f"an overall test accuracy of <b>{_pct(exp_ev['overall_accuracy'])}</b>."
    )
    rows = [["Class", "Expert", "Classifier", "Expert better?"]]
    for c in exp_ev["comparison"]:
        rows.append([c["cls"], _pct(c["expert"]), _pct(c["classifier"]),
                     "yes" if c["expert_better"] else "no"])
    S.append(_metric_table(rows, col_widths=[4 * cm, 3 * cm, 3 * cm, 3 * cm]))

    # 5. Learning to defer
    para("5. Learning-to-Defer Method", h1)
    para(
        "We use the Bayes-optimal deferral rule: defer when "
        "P(expert correct | x) &gt; max<sub>y</sub> P(y | x). P(expert correct | x) is "
        "learned from TF-IDF text features; the classifier's softmax confidence "
        "estimates its own correctness. We compare against a naive confidence-threshold "
        "baseline that ignores expert competence."
    )
    rows = [
        ["System", "Test accuracy"],
        ["Classifier only", _pct(adv["classifier_accuracy"])],
        ["Expert only", _pct(adv["expert_accuracy"])],
        ["Confidence-threshold team", _pct(conf["team_accuracy"])],
        ["Expert-advantage team", _pct(adv["team_accuracy"])],
        ["Oracle (upper bound)", _pct(adv["oracle_accuracy"])],
    ]
    S.append(_metric_table(rows, col_widths=[7 * cm, 4 * cm]))
    para(
        f"The expert-advantage team ({_pct(adv['team_accuracy'])}) beats the classifier "
        f"alone, the expert alone, and the confidence baseline. It defers "
        f"{_pct(adv['deferral_rate'])} of articles, with "
        f"{_pct(adv['useful_deferral_frac'])} useful and "
        f"{_pct(adv['harmful_deferral_frac'])} harmful deferrals.", body)

    # 6. Active learning
    para("6. Active Learning Method", h1)
    para(
        f"From this task the expert's labels are not available during training. We "
        f"query the expert on a pool of {act['pool_size']} training examples (budget "
        f"{act['budget']}) to learn P(expert correct | x). We compare margin "
        f"uncertainty sampling against a random baseline. Both strategies are "
        f"averaged over {act['n_runs']} runs and, within a run, start from the same "
        f"random warm-up set of {act['seed_size']} queries, so the gap between them "
        f"reflects the query strategy rather than luck in the starting labels."
    )
    rows = [["Expert queries", "Uncertainty", "Random"]]
    for u, r in zip(act["uncertainty_curve"], act["random_curve"]):
        rows.append([str(u["n_queries"]), _pct(u["team_accuracy"]), _pct(r["team_accuracy"])])
    S.append(_metric_table(rows, col_widths=[4 * cm, 4 * cm, 4 * cm]))
    para(
        f"Uncertainty sampling reaches the {act['target_accuracy']} team-accuracy "
        f"target in {act['uncertainty_queries_to_target']} queries versus "
        f"{act['random_queries_to_target']} for random, and leads at every budget. "
        f"Both approach the full-supervision ceiling "
        f"({_pct(act['full_supervision'])}, all pool labels).", body)

    # 7. Optional human interface
    para("7. Optional Human Expert Interface", h1)
    para(
        "The app also provides a demonstration interface in which a user acts as the "
        "expert: an article selected by uncertainty sampling is shown, the user picks "
        "a topic, and the label is stored. This illustrates how real human queries "
        "would feed the same active-learning pipeline used with the simulated expert."
    )

    # 8. Discussion
    para("8. Discussion", h1)
    para(
        "The human&ndash;AI team clearly surpasses both the classifier and the expert "
        "alone, confirming that learning to defer &mdash; using the expert's "
        "competence rather than mere classifier confidence &mdash; is beneficial. "
        "Active learning makes this efficient: uncertainty sampling concentrates the "
        "query budget on the deferral-relevant region, reaching strong performance "
        "with very few expert labels. Limitations include the simulated (rather than "
        "real) expert, a topic-based competence region, and a linear text model; a "
        "transformer classifier or a richer expert would be natural extensions."
    )

    # 9. Conclusion
    para("9. Conclusion", h1)
    para(
        f"The baseline reaches {_pct(base_ev['accuracy'])}; the simulated expert "
        f"{_pct(exp_ev['overall_accuracy'])}. Learning to defer lifts the team to "
        f"{_pct(adv['team_accuracy'])}, and active learning attains near that level "
        f"with only a few hundred expert queries. The human&ndash;AI team surpasses "
        f"the classifier baseline, meeting the project's goal."
    )

    doc.build(S)
    return buf.getvalue()
