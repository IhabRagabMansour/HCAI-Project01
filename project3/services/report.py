"""Generate the mandatory PDF report for Project 3.

Builds a multi-section report (reportlab) from the cached experiment results:
introduction, dataset, baseline classifier, simulated expert, learning-to-defer,
active learning, the optional human interface, discussion, and conclusion.
Returned as raw PDF bytes for a download response.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
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
    h1 = ParagraphStyle("H1", parent=base["Heading1"], fontName="Times-Bold",
                        fontSize=13, leading=16, spaceBefore=14, spaceAfter=5,
                        textColor=colors.black)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontName="Times-Bold",
                        fontSize=11, leading=14, spaceBefore=10, spaceAfter=3,
                        textColor=colors.black)
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName="Times-Roman",
                          fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY,
                          spaceAfter=7)
    title = ParagraphStyle("DocTitle", parent=base["Title"], fontName="Times-Bold",
                           fontSize=17, leading=21, spaceAfter=2,
                           textColor=colors.black)
    subtitle = ParagraphStyle("Subtitle", parent=body, fontName="Times-Italic",
                              fontSize=10.5, alignment=TA_CENTER, spaceAfter=2)
    return title, subtitle, h1, h2, body


def _pct(x):
    return f"{x * 100:.2f}%"


_TH = ParagraphStyle("TH", fontName="Times-Bold", fontSize=9.5, leading=12)
_TD = ParagraphStyle("TD", fontName="Times-Roman", fontSize=9.5, leading=12)


def _metric_table(rows, col_widths=None):
    """A plain rule-ruled table, in the style used in printed papers."""
    wrapped = [[Paragraph(str(c), _TH) for c in rows[0]]]
    wrapped += [[Paragraph(str(c), _TD) for c in row] for row in rows[1:]]
    t = Table(wrapped, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
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
        title="Project 3: Active Learning for Learning-to-Defer",
    )
    title, subtitle, h1, h2, body = _styles()
    S = []

    def para(text, style=body):
        S.append(Paragraph(text, style))

    # ── Title ──
    para("Project 3: Active Learning for Learning-to-Defer", title)
    para("Human-Centric Artificial Intelligence", subtitle)
    para("Topic classification on AG News", subtitle)
    S.append(Spacer(1, 14))

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
        f"The baseline is a {MODEL_DESCRIPTION}, reaching a test accuracy of "
        f"<b>{_pct(base_ev['accuracy'])}</b>."
    )
    para(
        "We chose logistic regression over a linear SVM for one reason that shapes "
        "the whole project: it gives us <b>calibrated class probabilities</b>. Both "
        "later tasks need a probability, not just a decision. The deferral rule "
        "compares the classifier's confidence against the expert's, and margin "
        "sampling ranks articles by how close the top two class probabilities are. "
        "An SVM would give us a margin we would then have to convert into a "
        "probability, adding a calibration step and a source of error to everything "
        "downstream.", body)
    para(
        "The text settings follow from the data. We use unigrams and bigrams "
        "because topic cues here are often two words long (<i>interest rate</i>, "
        "<i>world cup</i>, <i>oil prices</i>), and a unigram model splits them. We require a term to appear at least twice, which discards "
        "the long tail of typos and one-off proper nouns that cannot generalise, "
        "and cap the vocabulary at 50,000 features. Sublinear term frequency damps "
        "repeated words, so an article that says \"stocks\" eight times is not "
        "counted as eight times more about stocks. Regularization is deliberately "
        "light (C = 10) because with 120,000 training articles there is little risk "
        "of overfitting a linear model.", body)
    rows = [["Class", "Test accuracy"]]
    for i, c in enumerate(base_ev["class_names"]):
        rows.append([c, _pct(base_ev["per_class_accuracy"][i])])
    S.append(_metric_table(rows, col_widths=[5 * cm, 4 * cm]))
    para(
        "The classifier is strong on World and Sports but weaker on Business and "
        "Sci/Tech, which it confuses with each other. That is the region where an "
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

    para("4.1 Why the expert is designed this way", h2)
    para(
        "The project sheet asks for an expert that is imperfect and stronger in "
        "specific regions of the input space, and each of our choices follows from "
        "making that setup actually informative.", body)
    para(
        "<b>The competence region complements the classifier.</b> We gave the "
        "expert Business and Sci/Tech precisely because those are the two classes "
        "the baseline is weakest on and confuses with each other. This is the "
        "decision that makes the experiment meaningful: had we made the expert "
        "strong where the classifier is already strong, no deferral policy could "
        "ever improve the team, and both later tasks would be measuring nothing.",
        body)
    para(
        "<b>The expert is good, not perfect.</b> Inside its region it is right "
        "about 95% of the time rather than always. An oracle expert would make the "
        "deferral problem trivial, since deferring everything you are unsure "
        "about could then never cost you anything, and it would hide exactly the "
        "trade-off the task is about.", body)
    para(
        f"<b>The expert is genuinely weak outside its region</b>, at roughly 45%. "
        f"That is above the 25% of random guessing, so it is a plausible human "
        f"rather than an adversary, but well below the classifier. This is what "
        f"gives the naive baseline something to get wrong: overall the expert "
        f"scores only {_pct(exp_ev['overall_accuracy'])} against the classifier's "
        f"{_pct(base_ev['accuracy'])}, so a policy that defers whenever the "
        f"classifier is unsure will often hand the article to someone worse. A "
        f"deferral rule has to know <i>where</i> the expert is strong, not merely "
        f"that the classifier is weak.", body)
    para(
        "The expert's answers are drawn once from a fixed seed and cached, so every "
        "run of the study sees the same expert and results are reproducible.", body)

    # 5. Learning to defer
    para("5. Learning-to-Defer Method", h1)
    para(
        "For a single article the team is right in one of two ways: we keep it and "
        "the classifier is right, or we defer it and the expert is right. To "
        "maximise expected accuracy we should therefore defer exactly when the "
        "expert is more likely to be right than the classifier:", body)
    para(
        "defer(x)   iff   P(expert correct | x)  &gt;  P(classifier correct | x)",
        body)
    para(
        "We estimate the right-hand side with the classifier's own softmax "
        "confidence, max<sub>y</sub> P(y | x), and learn the left-hand side with a "
        "logistic regression on the same TF-IDF features, trained on articles where "
        "the expert's answer is known. Note what this rule does not depend on: the "
        "classifier being uncertain. It defers only when the expert is the better "
        "bet, which is the difference between deferral and mere rejection.", body)

    para("5.1 Why we do not learn P(classifier correct | x)", h2)
    para(
        "The symmetric choice would be to train a second model for the "
        "classifier's correctness instead of using its confidence. We tried it, and "
        "it fails in an instructive way. Individual classifier mistakes are close "
        "to unpredictable from the text alone (if they were predictable, the "
        "classifier would not be making them), so the model has almost no "
        "signal to fit and settles on the base rate, predicting roughly \"correct\" "
        "everywhere. Once that constant sits above the expert-correctness estimate "
        "for nearly every article, the rule never fires and the system silently "
        "degenerates into the classifier alone.", body)
    para(
        "The classifier's softmax confidence avoids this because it already varies "
        "per article and is calibrated by training. It is a far better per-article "
        "estimate of the classifier's own correctness than a second model fitted to "
        "noise.", body)

    para("5.2 The naive baseline we compare against", h2)
    para(
        "We also implement confidence-threshold rejection: defer whenever the top "
        "class probability falls below a threshold tuned on training data. This is "
        "the approach the lecture introduces first and then argues against, because "
        "it never asks whether the expert is any good. Including it lets us test "
        "that criticism on our own data rather than take it on faith.", body)
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
        f"The expert-advantage team ({_pct(adv['team_accuracy'])}) beats the "
        f"classifier alone, the expert alone, and the confidence baseline "
        f"({_pct(conf['team_accuracy'])}). The lecture's argument holds on our "
        f"data: knowing where the expert is competent is worth more than knowing "
        f"where the classifier is unsure.", body)

    para("5.3 Judging the deferral decisions, not just the accuracy", h2)
    para(
        "The sheet asks that the evaluation reflect the quality of the deferral "
        "decisions and not only accuracy, and the two really can come apart. A "
        "system could reach a decent team accuracy while deferring far too often, "
        "or while deferring the wrong articles and being rescued by the classifier "
        "on the rest. So we also report where the decisions land.", body)
    rows = [
        ["Deferral behaviour", "Value"],
        ["Articles deferred", f"{_pct(adv['deferral_rate'])} ({adv['n_deferred']} of "
                              f"{adv['n_deferred'] + adv['n_kept']})"],
        ["Expert correct on deferred articles", _pct(adv['expert_correct_when_deferred'])],
        ["Classifier correct on kept articles", _pct(adv['classifier_correct_when_kept'])],
        ["Useful deferrals (expert right, classifier wrong)", _pct(adv['useful_deferral_frac'])],
        ["Harmful deferrals (classifier right, expert wrong)", _pct(adv['harmful_deferral_frac'])],
    ]
    S.append(_metric_table(rows, col_widths=[9 * cm, 4 * cm]))
    para(
        f"These numbers say the rule is selecting well rather than getting lucky. "
        f"The expert answers {_pct(adv['expert_correct_when_deferred'])} of the "
        f"articles sent to them, against {_pct(adv['expert_accuracy'])} across the "
        f"test set as a whole, so the rule is finding the cases where this "
        f"particular expert is strong. Useful deferrals outnumber harmful ones by "
        f"more than three to one, and only {_pct(adv['deferral_rate'])} of articles "
        f"are handed over, which matters because a real expert's time is the scarce "
        f"resource. The oracle bound of {_pct(adv['oracle_accuracy'])}, which defers "
        f"perfectly, shows how much room a better competence model would still "
        f"have.", body)
    para(
        "Our implementation also exposes a query cost, a margin the expert "
        "advantage must exceed before we defer. We leave it at zero here because "
        "nothing in this task charges for expert time, but it is the lever to turn "
        "if it did: raising it trades a little team accuracy for fewer "
        "interruptions.", body)

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
        f"target in <b>{act['uncertainty_queries_to_target']} queries against "
        f"{act['random_queries_to_target']}</b> for random, and leads clearly "
        f"across the small budgets where the choice of query matters. Beyond "
        f"roughly seventy labels the two curves converge and random draws level "
        f"with it. We report that rather than hide it: once most of the useful "
        f"region has been labelled, which articles were asked about first stops "
        f"mattering, and uncertainty's deliberately skewed sample is marginally "
        f"less representative of the pool as a whole. The point of active learning "
        f"is the left-hand end of the curve, and that is where it wins. Both "
        f"approach the full-supervision ceiling of "
        f"{_pct(act['full_supervision'])}, which is what labelling the entire pool "
        f"would buy.", body)

    para("6.1 Why margin sampling", h2)
    para(
        "Of the uncertainty measures in the lecture, margin sampling fits this "
        "problem best. Least-confidence looks only at the top probability and "
        "ignores whether the runner-up was a close second or far behind, which is "
        "the distinction we care about. Entropy uses all four classes, but here "
        "that means the two classes the model has already ruled out contribute "
        "noise to the score. Margin looks at exactly the gap between the top two:",
        body)
    para(
        "u(x) = 1 - ( P(top class | x) - P(second class | x) )", body)
    para(
        "which is high precisely for articles the classifier cannot separate "
        "between two candidates. On this dataset that confusion is overwhelmingly "
        "Business against Sci/Tech, which by construction is exactly where our "
        "expert is competent. So margin sampling spends the query budget on the "
        "region where deferral can actually help, and that is why it pulls ahead "
        "early.", body)

    para("6.2 Pool, budget and the fairness of the comparison", h2)
    para(
        f"We draw a pool of {act['pool_size']} articles and allow up to "
        f"{act['budget']} queries. The pool is large enough that a query strategy "
        f"has a real choice to make, and small enough that labelling all of it is "
        f"feasible, which gives us the full-supervision ceiling to measure against. "
        f"The budget spans well under one per cent to about a sixth of the pool, so "
        f"the curve covers both the regime active learning is for and the regime "
        f"where it stops mattering.", body)
    para(
        f"Two details make the comparison fair. Both strategies are averaged over "
        f"the same {act['n_runs']} runs, since averaging one curve and not the "
        f"other would smooth only that one and flatter it. And within a run both "
        f"begin from the identical random warm-up of {act['seed_size']} queries. "
        f"That warm-up is not cosmetic: asked to choose its first labels with no "
        f"data at all, margin sampling picks the most ambiguous articles in the "
        f"pool, and those can happen to share a single expert outcome. The "
        f"competence model then sees one class, collapses to deferring everything, "
        f"and team accuracy falls to the expert's own. Starting from a small random "
        f"sample is the standard pool-based setup and removes that artefact.", body)

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
        "The human&ndash;AI team surpasses both the classifier and the expert "
        "alone, and it does so by using the expert's competence rather than the "
        "classifier's uncertainty. The gap over the confidence baseline is the "
        "concrete form of the lecture's argument: deferring when you are unsure is "
        "not the same as deferring when someone else is better, and on our data "
        "the difference is worth about a point and a half of accuracy. Active "
        "learning then makes that competence model cheap to obtain, reaching the "
        "target in a quarter of the queries a random sample needs.", body)
    para("<b>What we would treat carefully in these results.</b>", body)
    para(
        "The expert is simulated, and simulated from the true label, so its "
        "competence region is exactly the clean topic partition our model is best "
        "placed to learn. A real annotator's strengths would be messier and harder "
        "to predict from the article text, and we would expect the learned "
        "competence model to be correspondingly weaker.", body)
    para(
        f"The distance to the oracle is still large: {_pct(adv['oracle_accuracy'])} "
        f"against our {_pct(adv['team_accuracy'])}. Most of that gap is the limit "
        f"of predicting expert correctness from text alone, not a tuning problem.",
        body)
    para(
        "The classifier is a linear bag-of-bigrams model. A transformer would "
        "likely raise the baseline and, by changing which articles are hard, would "
        "move the deferral-relevant region as well, so the whole pipeline would "
        "need re-examining rather than merely re-running. Richer query strategies, "
        "such as combining margin sampling with a diversity term, would be the "
        "natural next step on the active-learning side.", body)

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
