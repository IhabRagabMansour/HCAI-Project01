# HCAI Project 04 Requirements

## Project title

**Project 4: Preference Elicitation**

## Goal

Design and implement a user study that compares two methods for eliciting movie preferences from a new user:

1. **Pairwise comparison:** repeatedly show two movies and ask which one the participant would rather watch.
2. **Ranking:** show ten movies and ask the participant to rank them from most preferred to least preferred.

The machine learning goal is to estimate a latent user preference vector `w` from a limited number of interactions. The human-centered goal is to design a rigorous experiment that determines which elicitation interface is more effective and/or efficient.

**Important:** You are **not required to conduct the user study**. You must design the complete protocol and implement an interface that could be used to run it.

## 1. Shared project requirements from previous projects

Project 4 should remain part of the same HCAI Django repository used for Projects 1 to 3.

### 1.1 General Django structure

- Implement Project 4 in Python using Django.
- Add Project 4 as a Django app inside the existing HCAI project.
- Make the app accessible from the shared home or launch page.
- Keep the project in the same Git repository as the earlier projects.
- Use project-specific templates and static files where useful.
- The interface does not need elaborate visual design, but it must be clear and usable by a study participant.

A reasonable app structure is:

```text
project4/
  __init__.py
  admin.py
  apps.py
  models.py
  urls.py
  views.py
  services/
    features.py
    preference_models.py
    study.py
  templates/
    project4/
      index.html
      consent.html
      instructions.html
      pairwise.html
      ranking.html
      questionnaire.html
      complete.html
  static/
    project4/
      style.css
```

Register the app in `settings.py` and include its URLs in the project-level `urls.py`.

Example:

```python
# settings.py
INSTALLED_APPS = [
    ...,
    "project4",
]
```

```python
# project-level urls.py
urlpatterns = [
    ...,
    path("project4/", include("project4.urls")),
]
```

The home page should contain a Project 4 link using the namespace chosen in your app.

## 2. Mandatory deliverables

### 2.1 Landing page

The Project 4 page must begin with a landing page from which the evaluator can:

1. **Download the PDF report.**
2. **Start the user study interface.**

This organization is explicitly required by the Project 4 sheet.

### 2.2 PDF report

A PDF report is mandatory and must be downloadable from the Project 4 interface.

The report must contain:

- the feature representation and extraction method from Task 1,
- the ranking extension of Bradley-Terry from Task 2,
- the complete user study design from Task 3,
- justification of important design choices.

Unlike Project 3, Project 4 does not ask for experimental results because the study does not have to be conducted.

### 2.3 Working study interface

The interface must be the interface that would actually be shown to a participant if the study were run.

At minimum, it must support both conditions:

- pairwise movie choices,
- ranking ten movies.

The interface should also implement the study flow needed by the protocol you describe in the report.

### 2.4 Source code

The final implementation must remain in the shared course Git repository with the other project apps.

## 3. Dataset requirements

Use the **IMDB 5000 Movie Dataset**.

The dataset contains metadata for approximately 5000 movies and does **not** contain user ratings or user preference labels.

This absence of ratings is central to the project: preferences have to be obtained directly from interactions with the participant.

### 3.1 Dataset role

For every movie, construct a feature vector:

```text
x in R^d
```

For every user, assume a latent preference vector:

```text
w in R^d
```

The project defines the utility of a movie for that user as:

```text
U(x) = w^T x
```

The elicitation process should collect preference observations and use them to estimate `w`.

## 4. Task 1 - Feature representation for movies

### Official requirement

Choose a feature representation that is appropriate for a movie recommender system.

You must:

- choose which movie attributes become features,
- justify the choice,
- implement a method that extracts the representation from the IMDB 5000 dataset.

The project sheet does **not** prescribe a particular representation.

### Recommended design principles

Because the user vector `w` must be estimated from relatively few interactions, prefer a feature representation that is:

- meaningful to human movie taste,
- not excessively high-dimensional,
- reasonably dense,
- consistently scaled,
- available for most movies.

### Recommended feature groups

A practical representation can include:

#### Genres

Represent genres as multi-hot binary features.

Example:

```text
Action = 1
Comedy = 0
Drama = 1
...
```

Genres are particularly suitable because they directly represent movie content and are easy to interpret as preference dimensions.

#### Numerical movie characteristics

Possible examples:

- duration,
- title year.

Standardize numerical features before fitting the preference model.

#### Moderate-cardinality categorical metadata

Possible examples:

- content rating,
- language,
- country.

Use one-hot encoding or a reduced set of common categories plus an `Other` category.

### High-cardinality features

Features such as actor names, director names, or plot keywords may produce a very large sparse vector. They can still be used, but limited interaction data makes estimating a large `w` difficult.

If you use them, justify dimensionality reduction or a top-K vocabulary strategy.

### Popularity and outcome features

Columns such as IMDB score, number of votes, gross revenue, or Facebook likes may encode general popularity rather than the participant's intrinsic taste.

The project does not prohibit using them. If you include or exclude them, explain why.

### Required preprocessing decisions to document

Your Task 1 section should state how you handle:

- missing values,
- categorical variables,
- multi-valued genre fields,
- numerical scaling,
- rare categories,
- final feature dimension `d`.

### Suggested feature extraction API

```python
def fit_movie_encoder(dataframe):
    """Fit preprocessing using the IMDB movie metadata."""


def transform_movies(dataframe, encoder):
    """Return the movie feature matrix X and movie metadata for display."""
```

The same representation must be used by both elicitation conditions.

## 5. Preference model for pairwise comparisons

For a movie `i`, define the latent score:

```text
s_i = w^T x_i
```

A standard Bradley-Terry probability for choosing movie `i` over movie `j` is:

```text
P(i > j | w)
    = exp(s_i) / (exp(s_i) + exp(s_j))
    = sigmoid(s_i - s_j)
    = sigmoid(w^T (x_i - x_j))
```

This model says that the larger the difference between the two latent scores, the more likely the user is to choose the higher-scoring movie.

### Estimating `w` from pairwise observations

For observed choices `(chosen_t, rejected_t)`, estimate `w` by maximizing the log-likelihood:

```text
L_pair(w)
    = sum_t log P(chosen_t > rejected_t | w)
```

A regularized version is recommended when only a few interactions are available:

```text
maximize L_pair(w) - lambda_reg * ||w||_2^2
```

Regularization is a recommended stability choice, not an explicit Project 4 requirement.

## 6. Task 2 - Extend Bradley-Terry to full rankings

### Official requirement

Propose an extension of Bradley-Terry that can model a complete ranking:

```text
i_1 > i_2 > ... > i_n
```

instead of only one pairwise comparison.

You must explain and justify the formulation.

### Recommended formulation: sequential Luce / Plackett-Luce ranking model

The cleanest extension follows the Luce finite-choice model.

For a displayed set of `n` movies with scores:

```text
s_i = w^T x_i
```

interpret a ranking as a sequence of choices:

1. choose the best item from all `n` movies,
2. remove it,
3. choose the best remaining item,
4. continue until the ranking is complete.

Then:

```text
P(i_1 > i_2 > ... > i_n | w)
  = product from k=1 to n-1 of
      exp(w^T x_{i_k})
      --------------------------------------------
      sum from j=k to n exp(w^T x_{i_j})
```

The final item has probability 1 once all other movies have been selected, so the last factor can be omitted.

### Why this is a good extension

- For `n = 2`, it reduces exactly to the Bradley-Terry pairwise probability.
- It uses the same latent score `w^T x` for both interfaces.
- It follows the Luce finite-choice model.
- It gives a valid likelihood for a complete ordered list.
- It allows `w` to be fitted directly from ranking data by maximum likelihood.

### Ranking log-likelihood

For one observed ranking:

```text
log P(ranking | w)
  = sum from k=1 to n-1 of
      [w^T x_{i_k} - logsumexp(w^T x_{i_k}, ..., w^T x_{i_n})]
```

For multiple ranking tasks, sum this quantity across all rankings.

A regularized objective can again be used:

```text
maximize L_rank(w) - lambda_reg * ||w||_2^2
```

### Alternative approach to discuss carefully

You could decompose one ranking into all implied pairwise comparisons and fit ordinary Bradley-Terry to those pairs.

For a ranking of 10 items this creates 45 pairwise relations.

This is simpler, but those pairwise relations come from the same ranking and are not independent observations. The sequential Luce formulation is therefore more principled as a direct ranking likelihood.

## 7. Fitting both preference models consistently

To compare the two interfaces fairly:

- use the same feature representation,
- use the same parameter dimension `w`,
- use comparable regularization,
- initialize optimization consistently,
- fit only from interactions collected in the corresponding condition.

Suggested functions:

```python
def fit_pairwise_model(pairwise_choices, X, regularization):
    ...


def fit_ranking_model(rankings, X, regularization):
    ...
```

The project does not require a specific optimizer. You may use SciPy optimization, PyTorch, or another suitable numerical method.

## 8. Task 3 - Complete user study design

### Official requirement

Design a complete user study comparing the two preference elicitation methods.

You must choose and justify at least:

- the research hypothesis,
- the experimental design,
- participant recruitment,
- study procedure,
- what data is collected,
- how the results would be analyzed,
- practical steps you would follow if the study were really conducted.

Again, you **do not have to run the study**.

## 8.1 Recommended research question

A strong research question is:

```text
Which elicitation interface learns a new user's movie preferences more efficiently:
pairwise choices or rankings of ten movies?
```

## 8.2 Recommended hypotheses

A concrete primary hypothesis could be:

```text
H1: For the same elicitation time budget, the ranking interface produces a preference
model with better predictive performance on held-out user choices than the pairwise
interface.
```

A secondary hypothesis could be:

```text
H2: Ranking ten movies creates greater subjective effort and longer interaction time per
task than a pairwise choice.
```

You may choose different hypotheses, but they must be measurable and connected to the two interfaces.

### Why equal time is a useful comparison

One ranking of ten movies contains much more preference information than one pairwise choice but also takes longer and may require more cognitive effort.

Therefore, comparing only an equal **number of tasks** may be misleading. A fair efficiency comparison can use one of these budgets:

- equal total interaction time,
- equal number of distinct movies inspected,
- equal total number of elicited preference relations.

Choose one primary budget and justify it in the report.

## 8.3 Independent and dependent variables

### Independent variable

The main experimental condition is:

```text
elicitation method in {pairwise, ranking}
```

### Primary dependent measures

Recommended objective measures include:

- held-out pairwise prediction accuracy,
- held-out negative log-likelihood or log-loss,
- time needed to reach a target predictive performance,
- predictive performance as a function of elapsed elicitation time.

### Secondary dependent measures

Recommended secondary measures include:

- task completion time,
- number of interactions completed,
- subjective ease of use,
- subjective mental effort,
- preference between the two interfaces.

The project sheet does not mandate specific metrics. The report must define and justify them.

## 8.4 Recommended experimental design

### Within-subject design

A strong default is a within-subject study:

- every participant completes both interfaces,
- the same participant therefore provides data for both methods,
- this controls for large individual differences in movie taste and decision behavior.

### Counterbalancing

Use two order groups:

```text
Group AB: pairwise first, ranking second
Group BA: ranking first, pairwise second
```

Assign participants randomly or as evenly as possible to these order groups.

### Avoiding direct carry-over

To reduce memory and carry-over effects:

- use disjoint elicitation movie sets for the two conditions,
- keep the movie sampling procedure otherwise equivalent,
- insert a short break between conditions,
- include practice trials before measurement begins.

The final held-out evaluation set can be shared across both learned models so that both are evaluated against the same participant judgments.

## 8.5 Recruitment plan

The project sheet requires you to choose a recruitment process but does not prescribe the population.

A reasonable target population is:

- adults who watch movies at least occasionally,
- able to understand the language used in the interface,
- able to use a standard browser interface.

Possible recruitment channels:

- university participant pool,
- student mailing lists,
- course/community recruitment,
- online participant platform, if institutionally permitted.

The report should define:

- inclusion criteria,
- exclusion criteria,
- target sample size and how it would be determined,
- compensation if applicable,
- recruitment channel.

Do not invent a mandatory participant count. The Project 4 sheet provides none.

## 8.6 Recommended study procedure

A complete practical protocol can be:

### Step 1: Information and consent

- Explain the study purpose at an appropriate level.
- Explain what interaction and demographic data will be stored.
- Explain withdrawal rights.
- Obtain informed consent.

### Step 2: Eligibility and background questionnaire

Collect only information that is useful for the study, for example:

- age range if needed,
- movie-watching frequency,
- familiarity with movie recommender systems.

Avoid unnecessary identifying data.

### Step 3: General instructions and practice

- Explain that choices should reflect what the participant would personally prefer to watch.
- Show a small number of practice tasks.
- Practice data should not be included in the main analysis.

### Step 4: First elicitation condition

Depending on counterbalanced order, run either:

- pairwise comparisons, or
- ten-movie ranking tasks.

Record:

- choices or rankings,
- movie IDs shown,
- timestamps or response times,
- condition order,
- task index.

### Step 5: Condition questionnaire

Collect subjective feedback such as:

- ease of expressing preferences,
- perceived effort,
- confidence that the interface captured their taste.

### Step 6: Break

Use a short break to reduce fatigue and carry-over.

### Step 7: Second elicitation condition

Run the other interface using an equivalent but non-overlapping elicitation movie set.

### Step 8: Second condition questionnaire

Collect the same subjective measures.

### Step 9: Common held-out preference evaluation

Show preference questions that were not used to fit either condition-specific model.

A common evaluation can use pairwise choices because both models can predict the probability of choosing one item over another.

Use the participant responses only for evaluation, not for refitting either model.

### Step 10: Final comparison and debrief

Ask which interface the participant preferred and why, then provide a debriefing.

## 8.7 Movie sampling

### Minimum required approach

The Project 4 sheet explicitly permits movies to be sampled **uniformly at random**.

This is sufficient for the required implementation.

### Practical constraints

When sampling, consider:

- avoid duplicate movies in the same ranking set,
- avoid presenting the same pair repeatedly unless intentional,
- decide whether to filter movies with insufficient metadata,
- use a reproducible random seed in development/testing,
- keep condition sampling rules equivalent.

### Optional adaptive extension

A stronger extension can choose movie queries based on how informative they are expected to be about `w`.

Examples:

- choose pairs whose predicted preference probability is near 0.5,
- choose movies that are diverse in feature space,
- choose ranking sets that cover dimensions of `w` that are still uncertain.

This is optional, not required.

## 8.8 Evaluation strategy

Because the true human `w` is unknown, do **not** make estimation error `||w_hat - w||` the primary human-study metric.

Instead, evaluate each learned `w` by how well it predicts new, held-out participant judgments.

### Pairwise held-out accuracy

For each held-out pair `(i, j)`, predict:

```text
choose i if P(i > j | w_hat) > 0.5
```

Then compare with the participant's actual choice.

### Held-out log-loss

Use the predicted probability rather than only the hard choice:

```text
-log P(observed choice | w_hat)
```

This is useful because it rewards calibrated preference probabilities.

### Efficiency curves

Since the project goal says the recommender should adapt **quickly**, evaluate predictive quality against:

- elapsed elicitation time,
- number of tasks,
- number of movies shown.

This lets you compare sample efficiency and interaction efficiency.

## 8.9 Planned statistical analysis

Because the recommended design is within-subject, the primary comparison is paired by participant.

The report should state:

1. the null hypothesis,
2. the primary outcome,
3. the statistical test you would use,
4. the significance level,
5. any correction if many hypotheses are tested.

Possible choices depend on the data distribution and metric. For example, a paired t-test can compare participant-level scores if its assumptions are reasonable; a non-parametric paired test can be chosen otherwise.

Hypotheses and exclusion criteria should be defined before looking at final results.

## 8.10 Data cleaning and exclusions

Predefine objective exclusion rules, such as:

- participant did not complete both required conditions,
- failed an attention or instruction check,
- implausibly fast repeated responses,
- technical failure caused missing interaction data.

Do not remove participants merely because their data does not support the hypothesis.

## 8.11 Ethics and privacy checklist

If the study were conducted:

- provide an information sheet and informed consent,
- allow withdrawal at any time without penalty,
- collect only required participant information,
- use pseudonymous participant IDs,
- store interaction logs securely,
- separate identifiers from preference data if identifiers are collected,
- define retention/deletion procedures,
- check institutional ethics requirements before recruitment,
- comply with applicable GDPR requirements.

## 9. Task 4 - Implement the participant interface

### Official requirement

Implement the interface that would be given to a participant in the study.

The interface must support the study design described in Task 3.

## 9.1 Landing page requirements

The Project 4 landing page should contain at least:

- project title and short explanation,
- **Download report** button,
- **Start study** button.

## 9.2 Pairwise condition

For each pairwise task, display two movies side by side.

Each movie should show enough information for a meaningful preference decision, for example:

- title,
- year,
- genres,
- optional poster if legally and technically available,
- optional short metadata such as duration or main actors.

The participant should be able to select one movie and continue.

Recommended interface behavior:

- do not preselect an answer,
- prevent continuing until a valid choice is made,
- show simple progress information,
- record response time,
- avoid showing model predictions because they could influence the participant.

## 9.3 Ranking condition

Display exactly **ten movies** for the ranking condition, as required by the project sheet.

The participant must be able to order them from most preferred to least preferred.

A practical implementation can use:

- drag and drop,
- ranked slots from 1 to 10,
- move up / move down buttons as an accessible fallback.

The final ranking must contain every displayed movie exactly once.

Before submission:

- validate that all 10 positions are filled,
- prevent duplicates,
- clearly indicate that rank 1 means most preferred.

## 9.4 Study state

The interface should track at least:

```text
participant/session ID
condition order
current condition
current task index
movies shown
responses/rankings
timestamps or response times
questionnaire responses
study completion status
```

Use Django sessions or database models depending on your architecture.

## 9.5 Recommended Django models

If you persist study data, possible models include:

```text
StudySession
PairwiseTrial
RankingTrial
QuestionnaireResponse
```

Example conceptual fields:

```text
StudySession:
  id
  anonymous_participant_code
  condition_order
  started_at
  completed_at

PairwiseTrial:
  session
  task_index
  movie_left_id
  movie_right_id
  chosen_movie_id
  response_time_ms

RankingTrial:
  session
  task_index
  movie_ids
  ranked_movie_ids
  response_time_ms
```

The exact database design is your choice.

## 9.6 Study progress and stopping rule

The Project 4 sheet does not specify:

- number of pairwise trials,
- number of ranking trials,
- time budget,
- stopping rule.

You must choose these as part of Task 3 and make the interface consistent with that choice.

A good report explains why the selected budget gives a meaningful comparison without excessive participant fatigue.

## 10. Recommended model update behavior in the interface

The study can estimate `w` after every response or after each block.

A useful implementation is:

```text
1. Present task.
2. Record preference response.
3. Append response to current condition data.
4. Refit or update w for that condition.
5. Select the next movie query.
6. Continue until the stopping criterion is reached.
```

If movie selection is uniformly random, step 5 is simple random sampling.

Do not display the current `w` or recommendations to participants unless your experimental protocol explicitly includes that information, because feedback from the model could change subsequent preferences and become a confound.

## 11. Recommended report structure

## 11.1 Introduction

Explain:

- preference elicitation,
- cold-start motivation for a new movie recommender user,
- why comparing pairwise and ranking interfaces is useful.

## 11.2 Dataset

Describe:

- IMDB 5000 Movie Dataset,
- lack of user ratings,
- metadata used for features,
- filtering and missing-value handling.

## 11.3 Task 1 - Feature representation

Include:

- exact selected columns,
- encoding method,
- scaling,
- final dimensionality,
- justification,
- feature extraction implementation.

## 11.4 Pairwise preference model

Include:

- `U(x) = w^T x`,
- Bradley-Terry probability,
- likelihood,
- method used to estimate `w`,
- regularization if used.

## 11.5 Task 2 - Ranking model

Include:

- complete mathematical formula,
- explanation as repeated selection from remaining items,
- demonstration that the two-item case reduces to Bradley-Terry,
- ranking likelihood and estimation method.

## 11.6 Task 3 - Research question and hypotheses

State:

- research question,
- primary hypothesis,
- null hypothesis,
- secondary hypotheses if used.

## 11.7 Experimental design

Describe:

- within-subject, between-subject, or mixed design,
- condition order,
- randomization,
- counterbalancing,
- elicitation budget,
- movie sampling,
- held-out evaluation.

## 11.8 Participants and recruitment

Describe:

- target population,
- inclusion criteria,
- exclusion criteria,
- recruitment method,
- compensation,
- sample-size planning.

## 11.9 Procedure

Give the full chronological procedure that would be followed in practice.

## 11.10 Measures

Separate:

- objective model-performance measures,
- interaction measures,
- subjective measures.

## 11.11 Planned analysis

Explain:

- how data would be cleaned,
- primary statistical comparison,
- descriptive statistics and plots,
- handling of multiple hypotheses if applicable.

## 11.12 Ethics and data protection

Explain:

- consent,
- withdrawal,
- stored data,
- pseudonymization/anonymization,
- secure storage,
- planned retention/deletion.

## 11.13 Interface implementation

Describe:

- landing page,
- pairwise interface,
- ranking interface,
- study state and logging,
- how the interface matches the planned protocol.

## 11.14 Limitations and extensions

Possible limitations:

- linear utility model may not capture complex tastes,
- stated preference may differ from actual movie-watching behavior,
- ranking 10 movies can be cognitively demanding,
- random movie selection may be sample-inefficient,
- unfamiliar movies can make choices noisy.

Possible extension:

- adaptive query selection based on information gain or uncertainty.

## 12. Full interface checklist

### Landing page

- [ ] Project 4 title
- [ ] Short project description
- [ ] Downloadable PDF report
- [ ] Start study button

### Study preparation

- [ ] Participant information page
- [ ] Consent flow in the planned protocol
- [ ] Instructions
- [ ] Practice task(s)

### Pairwise interface

- [ ] Exactly two movies shown per trial
- [ ] Clear movie information
- [ ] Participant chooses one movie
- [ ] No accidental duplicate movie in a pair
- [ ] Progress indicator
- [ ] Response logged
- [ ] Response time logged if used by the analysis

### Ranking interface

- [ ] Exactly ten movies shown per ranking task
- [ ] Participant can rank all ten
- [ ] Rank direction is explicit
- [ ] Every movie appears exactly once in submitted ranking
- [ ] Progress indicator
- [ ] Ranking logged
- [ ] Response time logged if used by the analysis

### Study design integration

- [ ] Condition assignment/order implemented
- [ ] Counterbalancing implemented if using within-subject design
- [ ] Break or transition screen implemented if included in protocol
- [ ] Condition questionnaires implemented if included
- [ ] Held-out evaluation implemented if included
- [ ] Final debrief/completion page

## 13. Task-by-task acceptance checklist

## Task 1

- [ ] Feature representation chosen
- [ ] Representation justified
- [ ] Extraction/preprocessing code implemented
- [ ] Missing values handled
- [ ] Categorical features encoded
- [ ] Numerical features scaled where appropriate
- [ ] Final feature dimension documented

## Task 2

- [ ] Standard Bradley-Terry model stated
- [ ] Ranking extension defined mathematically
- [ ] Extension justified
- [ ] Probability of a full ranking computable
- [ ] Two-item case reduces to Bradley-Terry
- [ ] `w` can be estimated from ranking data

## Task 3

- [ ] Research question defined
- [ ] Testable hypothesis defined
- [ ] Independent and dependent variables defined
- [ ] Experimental design chosen and justified
- [ ] Recruitment process defined
- [ ] Inclusion/exclusion criteria defined
- [ ] Elicitation budget/stopping rule defined
- [ ] Movie sampling strategy defined
- [ ] Practical procedure described step by step
- [ ] Ethics/consent/privacy addressed
- [ ] Analysis plan defined
- [ ] Pilot-study plan described

## Task 4

- [ ] Participant-facing study interface implemented
- [ ] Pairwise condition works
- [ ] Ten-movie ranking condition works
- [ ] Study order/state is tracked
- [ ] Responses are stored or logged consistently
- [ ] Landing page links to study
- [ ] Landing page provides PDF report download

## 14. Minimum viable implementation

A minimum solid submission should contain:

1. Project 4 Django app linked from the course home page.
2. IMDB 5000 data loader and feature extractor.
3. Pairwise Bradley-Terry model.
4. Ranking likelihood extending Bradley-Terry.
5. Method to estimate `w` for both interaction types.
6. Complete written user-study protocol.
7. Working pairwise participant interface.
8. Working ten-movie ranking participant interface.
9. Landing page with **Start study** and **Download report** actions.
10. PDF report describing Tasks 1 to 3 and design choices.

## 15. Stronger implementation ideas

These are optional improvements, not requirements:

- update `w` live after each interaction,
- visualize model convergence on a developer-only page,
- adaptive movie selection using uncertainty or information gain,
- diversity-aware query selection,
- reproducible study configuration stored in a database,
- export anonymized study logs to CSV,
- accessibility-friendly ranking controls in addition to drag and drop,
- server-side validation of ranking uniqueness,
- automatic generation of condition order,
- simulation mode for testing the study without recruiting participants,
- compare MLE with Bayesian estimation of `w`.

## 16. Important distinctions

### Required versus recommended ranking model

The project requires **an extension of Bradley-Terry to rankings**, but it does not name the extension.

The sequential Luce / Plackett-Luce formulation in this document is a recommended solution derived naturally from the probabilistic user model.

### Required versus optional query selection

Uniform random movie selection is explicitly allowed and is sufficient.

Adaptive or active movie selection is only an extension.

### Required versus optional user-study execution

You must design the study and implement the interface.

You do **not** have to recruit participants or collect results.

### Required versus optional participant count

No mandatory sample size is specified in the Project 4 sheet.

Choose and justify a sample-size planning method in the protocol rather than inventing a course requirement.

### Required versus optional evaluation metrics

The project does not prescribe a single metric for comparing the two interfaces.

Your report must define metrics that match your hypothesis. Held-out predictive performance and interaction efficiency are strong choices.

## 17. Common mistakes to avoid

- Treating the IMDB 5000 dataset as if it already contains user ratings.
- Using a different movie feature representation for the two conditions.
- Calling a ranking simply ten independent pairwise observations without discussing dependence.
- Failing to define how `w` is estimated from responses.
- Comparing conditions only on training likelihood instead of held-out preferences.
- Using `||w_hat - w||` as a human-study metric when the true human `w` is unknown.
- Using an equal number of tasks without considering that a ten-item ranking contains more information and takes more effort than one pairwise choice.
- Using a within-subject design without counterbalancing condition order.
- Showing model recommendations during elicitation without accounting for the fact that this may influence later responses.
- Collecting unnecessary identifiable participant data.
- Omitting consent, withdrawal, or data-protection considerations from the planned protocol.
- Forgetting that the ranking interface must contain **ten movies**.
- Forgetting the required landing page with both **report download** and **start study** actions.
- Reporting made-up user-study results even though the project only asks for the protocol.

## 18. Recommended implementation order

1. Load and inspect IMDB 5000 data.
2. Choose and implement the movie feature representation.
3. Implement the Bradley-Terry pairwise probability.
4. Implement pairwise MLE for `w`.
5. Implement the ranking likelihood.
6. Implement ranking MLE for `w`.
7. Add unit tests showing that a two-item ranking matches Bradley-Terry.
8. Design the complete study protocol and freeze study parameters in a configuration object.
9. Implement the Project 4 landing page.
10. Implement pairwise participant trials.
11. Implement ten-movie ranking trials.
12. Implement condition assignment and study state.
13. Add questionnaires/evaluation steps required by your protocol.
14. Add secure response logging.
15. Write the PDF report.
16. Add the report download button.
17. Test the complete study from start to finish using a fresh session.

## 19. Suggested source-code modules

```text
project4/
  services/
    data.py
      load_imdb_movies()
      clean_movies()

    features.py
      fit_movie_encoder()
      transform_movies()

    preference_models.py
      pairwise_probability()
      pairwise_log_likelihood()
      ranking_log_likelihood()
      fit_pairwise_w()
      fit_ranking_w()

    sampling.py
      sample_pair()
      sample_ranking_set()
      optional_adaptive_query()

    study.py
      assign_condition_order()
      next_trial()
      validate_response()
      study_progress()
```

Keep mathematical code separate from Django view code so that the preference model can be unit-tested independently from the interface.

## 20. Suggested tests

### Mathematical tests

- [ ] Pairwise probabilities sum correctly.
- [ ] `P(i > j) + P(j > i) = 1`.
- [ ] Two-item ranking probability equals Bradley-Terry.
- [ ] Ranking probability is between 0 and 1.
- [ ] Ranking log-likelihood is finite for valid data.
- [ ] Fitting synthetic preferences approximately recovers a known synthetic `w`.

### Interface tests

- [ ] Pairwise trial cannot submit without a choice.
- [ ] Ranking trial cannot submit with missing movies.
- [ ] Ranking trial cannot submit duplicate positions.
- [ ] Study order persists across page reloads if intended.
- [ ] Report download works.
- [ ] Fresh session starts at the correct first page.
- [ ] Study can reach the completion/debrief page.

## 21. Final submission checklist

- [ ] Project 4 is linked from the shared home page.
- [ ] Landing page has report download and start-study actions.
- [ ] PDF report is present and downloadable.
- [ ] Task 1 implementation and justification are in the report.
- [ ] Task 2 formulation and justification are in the report.
- [ ] Task 3 complete user-study design is in the report.
- [ ] Task 4 participant interface is implemented.
- [ ] Pairwise condition works.
- [ ] Ranking condition displays ten movies and works.
- [ ] Movie feature extraction is reproducible.
- [ ] `w` estimation is implemented for pairwise data.
- [ ] `w` estimation is implemented for ranking data.
- [ ] Study order and progress are handled.
- [ ] Participant data handling matches the written protocol.
- [ ] No fabricated study results are presented as real data.
- [ ] Source code is committed to the shared Git repository.

## 22. Source documents used

This requirements guide was prepared from the attached course materials:

- `HCAI-project_04.pdf` - official Project 4 specification.
- `HCAI-project_01.pdf`, `HCAI-project_02.pdf`, and `HCAI-project_03.pdf` - shared Django/project conventions and deliverable continuity.
- `README.md`, `HCAI_project_02.md`, and `HCAI_project_03.md` - style and structure of the existing project documentation.
