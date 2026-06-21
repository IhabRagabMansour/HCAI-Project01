# HCAI Project 03 Requirements

## Project title

**Project 3: Active Learning for Learning-to-Defer**

## Goal

Implement active learning methods for human-AI collaboration through learning-to-defer.

The system should train a classifier that can either:

1. Make a prediction itself.
2. Defer the decision to a human expert.

The project focuses on learning when deferral is useful, especially when expert labels are expensive and must be actively queried.

## Shared project requirements from previous projects

Project 3 should follow the same overall structure as the previous HCAI projects.

### General structure

- Implement the project in Python using Django.
- Add Project 3 as an app inside the existing HCAI Django project.
- Make the app accessible from the launch page or home page.
- Add a link to Project 3 in the home page project list.
- Keep the project inside the same Git repository as the other projects.
- Use a reasonable web interface. The focus is not UI design, but the app should be usable.
- If plots or figures are generated with matplotlib, save them into the media directory and display the generated images in Django, or use a JavaScript plotting library such as ChartJS.
- Use project-specific templates and static files where needed.

### Suggested Django app setup

The app can be named `project3`.

Expected structure:

```text
project3/
  __init__.py
  admin.py
  apps.py
  models.py
  urls.py
  views.py
  templates/
    project3/
      index.html
  static/
    project3/
      style.css
```

The app should be registered in `settings.py`:

```python
INSTALLED_APPS = [
    ...
    "project3",
]
```

The project-level `urls.py` should include the app:

```python
from django.urls import include, path

urlpatterns = [
    ...
    path("project3/", include("project3.urls")),
]
```

The app-level `urls.py` should define an index route:

```python
from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
]
```

The home page should include a link similar to:

```python
{"name": "Project 3", "url_name": "project3:index"}
```

## Mandatory deliverables

### 1. Working Django interface

The Project 3 interface should allow the user to access the main parts of the project:

- Dataset and baseline model information.
- Simulated expert information.
- Learning-to-defer results.
- Active learning experiment results.
- Optional human expert interaction, if implemented.
- A downloadable PDF report.

### 2. PDF report

A PDF report is mandatory.

The report must be accessible from the Project 3 interface, for example through a download button.

The report must contain:

- A description of the experiments.
- A justification of the design choices.
- A detailed report of the results.
- Metrics used for evaluation.
- Discussion of the classifier, expert, deferral strategy, and active learning strategy.

### 3. Source code

Submit the implementation as part of the shared Git repository used for the HCAI projects.

## Dataset requirements

Use the **AG News** dataset:

```text
https://huggingface.co/datasets/fancyzhx/ag_news
```

The task is topic classification for news articles.

### Dataset usage

- Use the training split for training.
- Use the test split only for evaluation.
- Do not use the test set for model selection or active learning query selection.
- The classifier should predict the topic label of each news article.

### Suggested preprocessing requirements

The project statement does not prescribe a specific text preprocessing pipeline, so this is a design choice. The report should justify the chosen approach.

Possible choices include:

- TF-IDF vectorization with a linear classifier.
- Bag-of-words vectorization with logistic regression or linear SVM.
- A pretrained text embedding model with a classifier.
- A transformer-based classifier, if computationally feasible.

The minimum requirement is that the chosen model works on the AG News text classification task and produces test accuracy.

## Task 1: Baseline classifier

### Requirement

Propose a classification model for AG News.

Train the model using all available labels in the training set.

Report the test accuracy.

### Expected implementation

The implementation should:

- Load the AG News dataset.
- Extract the article text and labels.
- Preprocess or vectorize the text.
- Train a classification model using the full labeled training set.
- Evaluate the trained model on the test set.
- Report test accuracy.

### Expected interface output

The interface should display:

- The chosen model type.
- Important preprocessing choices.
- The number of training examples used.
- The number of test examples used.
- Test accuracy.
- Any additional relevant metrics, if implemented.

### Expected report content

The report should explain:

- Why the model was selected.
- How the text was represented.
- How the training and evaluation were performed.
- The resulting test accuracy.
- Whether the baseline is strong enough to act as a target for the human-AI team.

## Task 2: Simulated expert

### Requirement

Design and implement at least one simulated expert.

The expert should not be perfect.

The expert should exhibit expertise in specific regions of the input space.

The expert's predictions should be available when queried.

Analyze the expert's strengths and weaknesses on the dataset.

Report the expert's accuracy on the test set.

### Expected implementation

The implementation should include at least one expert simulation function or class.

The expert should:

- Receive an input article.
- Return a predicted label.
- Be more accurate in some regions of the input space.
- Be less accurate in other regions.
- Avoid behaving like a perfect oracle.

### Possible expert designs

The exact design is open, but it must be justified. Examples:

1. **Topic specialist expert**

   The expert is highly accurate for one or two AG News topics and weaker for the others.

2. **Keyword-based expert**

   The expert performs well when topic-specific keywords are present and poorly otherwise.

3. **Confidence-based expert**

   The expert is accurate when the baseline classifier is confident or when examples are close to known regions, and inaccurate elsewhere.

4. **Length-based or ambiguity-based expert**

   The expert performs better on short, clear articles and worse on long or ambiguous articles.

5. **Noisy expert with region-specific competence**

   The expert uses the true label with high probability in its competence region and returns noisy or biased labels outside that region.

### Expected interface output

The interface should display:

- Description of the simulated expert.
- Expert accuracy on the test set.
- Strengths of the expert.
- Weaknesses of the expert.
- Examples where the expert performs well.
- Examples where the expert fails, if implemented.

### Expected report content

The report should explain:

- How the expert was simulated.
- Why this simulation represents region-specific expertise.
- Where the expert is strong.
- Where the expert is weak.
- Expert test accuracy.
- Any class-wise or region-wise analysis used to support the design.

## Task 3: Learning to defer

### Requirement

Implement a learning-to-defer strategy.

For each input, the system must choose between:

1. Producing a prediction itself.
2. Querying the expert.

Report the performance of the trained system.

The evaluation must reflect:

- Classification accuracy.
- Quality of the deferral decisions.

### Training setting

For Task 3, both classifier labels and expert labels are available.

This means the system can learn when the expert is better than the classifier.

### Expected implementation

The implementation should include:

- A classifier prediction function.
- An expert prediction function.
- A deferral decision rule or deferral model.
- A combined human-AI prediction rule.

For an input `x`:

```text
if defer(x) == True:
    prediction = expert(x)
else:
    prediction = classifier(x)
```

### Possible deferral strategies

The project does not prescribe a specific method. The strategy must be justified.

Possible approaches:

1. **Confidence thresholding**

   Defer when the classifier confidence is below a threshold.

2. **Error predictor**

   Train a model to predict whether the classifier is likely to be wrong.

3. **Expert advantage predictor**

   Train a model to predict whether the expert is more likely to be correct than the classifier.

4. **Cost-sensitive deferral**

   Defer only when the expected gain from asking the expert exceeds a chosen cost.

5. **Reject-option classifier**

   Treat deferral as an additional output decision.

### Required evaluation

At minimum, report classification accuracy of the combined system.

The evaluation should also measure deferral quality.

Recommended metrics:

- Classifier-only accuracy.
- Expert-only accuracy.
- Human-AI team accuracy.
- Deferral rate.
- Accuracy on deferred examples.
- Accuracy on non-deferred examples.
- Fraction of useful deferrals.
- Fraction of harmful deferrals.
- Expert correctness when deferred.
- Classifier correctness when not deferred.
- Comparison against an oracle deferral policy, if implemented.

### Expected interface output

The interface should display:

- Selected deferral strategy.
- Classifier-only performance.
- Expert-only performance.
- Human-AI team performance.
- Deferral rate.
- Deferral quality metrics.
- Example deferred and non-deferred articles, if implemented.

### Expected report content

The report should explain:

- The deferral strategy.
- What information is used to decide whether to defer.
- How the deferral model is trained.
- How the final prediction is produced.
- How performance is evaluated.
- Whether deferral improves over the classifier baseline.
- Whether deferral improves over using the expert alone.
- Whether deferral decisions are sensible.

## Task 4: Active learning for expert competence discovery

### Requirement

Choose an active learning strategy for querying the expert.

The goal is to efficiently learn when deferral is beneficial.

Justify the chosen strategy.

Report the results using metrics of your choice.

### Training setting

From Task 4 onward, assume that expert labels are not available during training.

The algorithm:

- Has access to the whole training dataset to learn the classifier.
- Does not initially have access to any expert data.
- Can query the expert when relevant to get a new expert label on a specific data point.

### Meaning of the task

The system must actively decide which examples should be shown to the expert.

The purpose of querying is to learn:

1. The classification task.
2. The expert's competence profile.
3. When deferral is beneficial.

### Expected implementation

The implementation should include:

- A query strategy.
- A query budget or stopping rule.
- A loop for querying the simulated expert.
- An updated deferral model trained from queried expert labels.
- Evaluation after different numbers of expert queries.

A typical loop:

```text
1. Train classifier using available training labels.
2. Start with no expert labels.
3. Select an unlabeled example to query from the expert.
4. Store the expert's answer.
5. Update the expert competence model or deferral model.
6. Repeat until the query budget is exhausted.
7. Evaluate the final human-AI system on the test set.
```

### Possible active learning strategies

The project does not prescribe a specific method. The selected method must be justified.

Possible strategies:

1. **Uncertainty sampling**

   Query examples where the classifier is least confident.

2. **Disagreement sampling**

   Query examples where models disagree or where the classifier and estimated expert behavior are uncertain.

3. **Expected value of deferral**

   Query examples that are most informative for deciding whether the expert or classifier is better.

4. **Diversity-based sampling**

   Query examples that cover different parts of the input space.

5. **Class-balanced sampling**

   Query examples across all topic classes to avoid learning an expert profile biased toward frequent classes.

6. **Hybrid strategy**

   Combine uncertainty, diversity, and predicted expert benefit.

### Required evaluation

The report should include metrics that show whether active learning efficiently learns useful deferral behavior.

Recommended metrics:

- Human-AI team accuracy as a function of number of expert queries.
- Deferral rate as a function of number of expert queries.
- Deferral precision, meaning how often deferred examples are actually better handled by the expert.
- Number of expert queries needed to reach a target performance.
- Comparison with random querying.
- Comparison with using all expert labels, if implemented.
- Comparison with no deferral.
- Accuracy improvement per expert query.
- Expert competence prediction accuracy, if implemented.

### Expected interface output

The interface should display:

- Selected active learning strategy.
- Query budget.
- Number of expert queries used.
- Human-AI performance after querying.
- Performance curve over query rounds, if implemented.
- Comparison to random querying, if implemented.
- Examples selected for expert queries, if implemented.

### Expected report content

The report should explain:

- Why the active learning strategy was selected.
- What the query budget was.
- How examples were selected for expert querying.
- How the deferral model was updated.
- What metrics were used.
- Whether active learning was more efficient than random querying.
- How quickly the system learned the expert's competence profile.
- Limitations of the approach.

## Task 5: Optional active learning with human expert

### Requirement

This task is optional.

Implement an interface where a user can interact with the system and provide the requested labels.

### Expected implementation if included

The interface may allow a user to:

- View a selected article.
- Provide the label they think is correct.
- Submit their answer.
- Store the user-provided label.
- Use the collected human labels in the active learning or deferral pipeline.

### Expected interface output if included

The interface may include:

- Article text.
- Label options.
- Submit button.
- Current query number.
- Query budget progress.
- Feedback after submission, if appropriate.
- Updated model or experiment results.

### Expected report content if included

The report should explain:

- How the human interaction was implemented.
- How user labels are collected and stored.
- Whether these labels are used only for demonstration or for actual training.
- How this differs from the simulated expert setup.

## Full interface checklist

The Project 3 app should ideally contain the following regions or pages.

### 1. Overview page

- Project title.
- Short explanation of learning-to-defer.
- Dataset information.
- Navigation to baseline, expert, deferral, active learning, and report sections.

### 2. Baseline classifier section

- Model description.
- Training configuration.
- Test accuracy.
- Additional metrics, if available.

### 3. Simulated expert section

- Expert description.
- Expert test accuracy.
- Strength and weakness analysis.
- Optional examples.

### 4. Learning-to-defer section

- Deferral strategy.
- Human-AI team performance.
- Deferral rate.
- Deferral quality metrics.
- Optional examples of deferred decisions.

### 5. Active learning section

- Active learning strategy.
- Query budget.
- Query selection method.
- Performance over query rounds.
- Comparison to random querying, if available.

### 6. Optional human expert section

- Interactive article labeling.
- User label submission.
- Query progress display.

### 7. Report download section

- A visible button or link to download the PDF report.

## Report checklist

The PDF report should include the following sections.

### 1. Introduction

- Problem statement.
- Why learning-to-defer is useful.
- Overview of the AG News task.
- Overview of the system.

### 2. Dataset

- Dataset name and source.
- Train and test split usage.
- Label prediction task.
- Any preprocessing steps.

### 3. Baseline classifier

- Model choice.
- Feature representation.
- Training setup.
- Test accuracy.
- Discussion of baseline strength.

### 4. Simulated expert

- Expert design.
- Competence regions.
- Noise or imperfection mechanism.
- Strengths and weaknesses.
- Expert test accuracy.

### 5. Learning-to-defer method

- Deferral strategy.
- Training data used.
- Deferral decision rule.
- Metrics.
- Results.
- Discussion of deferral quality.

### 6. Active learning method

- Starting assumption of no expert labels.
- Query strategy.
- Query budget.
- Update process.
- Metrics.
- Results over query rounds.
- Comparison to baseline query strategy, if implemented.

### 7. Optional human expert interface

- Interface description.
- Label collection mechanism.
- How labels are used.

### 8. Discussion

- What worked well.
- What failed.
- Limitations.
- Possible improvements.

### 9. Conclusion

- Final summary of classifier, expert, deferral, and active learning results.
- Whether the human-AI team matched or surpassed the baseline.

## Minimum viable implementation

A minimum acceptable implementation should include:

- A Django Project 3 app linked from the home page.
- Loading AG News.
- A baseline text classifier trained on the full labeled training set.
- Baseline test accuracy.
- At least one imperfect simulated expert.
- Expert accuracy and strength or weakness analysis.
- A learning-to-defer strategy.
- Human-AI team evaluation with accuracy and deferral quality metrics.
- An active learning query strategy for collecting expert labels.
- Results showing performance under the active learning setup.
- A PDF report downloadable from the interface.

## Stronger implementation ideas

These are not strictly required, but they can improve the project.

- Compare multiple classifiers.
- Compare multiple simulated experts.
- Compare multiple deferral strategies.
- Compare active learning against random querying.
- Plot performance versus number of expert queries.
- Add class-wise accuracy analysis.
- Add confusion matrices.
- Add examples of correct and incorrect deferrals.
- Add a real human labeling interface.
- Add query budget controls in the UI.
- Add downloadable experiment results.

## Important distinctions

### Baseline classification

The classifier predicts the topic directly.

### Simulated expert

The expert provides predictions when queried, but is imperfect.

### Learning to defer

The system learns when to use the classifier and when to use the expert.

### Active learning

The system does not initially know the expert's labels. It chooses which examples to query in order to learn the expert's competence efficiently.

## Final acceptance checklist

Use this checklist before submission.

- [ ] Project 3 app exists in the Django project.
- [ ] Project 3 is registered in `INSTALLED_APPS`.
- [ ] Project 3 has its own `urls.py`.
- [ ] Project 3 is included in the main URL configuration.
- [ ] Project 3 is linked from the home page.
- [ ] AG News dataset is loaded correctly.
- [ ] Train split is used for training.
- [ ] Test split is used only for evaluation.
- [ ] Baseline classifier is trained on all available training labels.
- [ ] Baseline test accuracy is reported.
- [ ] At least one simulated expert is implemented.
- [ ] Simulated expert is imperfect.
- [ ] Simulated expert has region-specific strengths.
- [ ] Expert test accuracy is reported.
- [ ] Expert strengths and weaknesses are analyzed.
- [ ] Learning-to-defer strategy is implemented.
- [ ] The system can choose between classifier prediction and expert query.
- [ ] Human-AI team performance is reported.
- [ ] Deferral quality is evaluated.
- [ ] Active learning query strategy is implemented.
- [ ] Active learning starts without expert labels.
- [ ] The system can query the expert for selected training examples.
- [ ] Results are reported with appropriate metrics.
- [ ] Design choices are justified.
- [ ] PDF report is generated.
- [ ] PDF report is accessible from the interface.
- [ ] Code is included in the shared Git repository.

## Source documents used

- HCAI-project_03.pdf
- HCAI-project_01.pdf
- HCAI-project_02.pdf
