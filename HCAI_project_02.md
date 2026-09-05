# HCAI Project 2: Explainability

## 1. Project purpose

The project is about building an interactive explainability dashboard for machine learning models trained on the Palmer Penguins dataset. The dashboard must let a user compare model performance and model complexity, generate counterfactual explanations, and inspect global feature effects using PDP and ALE plots.

The central idea is the interpretability objective:

```text
minimize empirical loss + lambda * model complexity
```

In the project sheet this is written as:

```text
arg min_f (1/n) sum_i loss(f(x_i), y_i) + lambda * Omega(f)
```

where:

- `loss` measures prediction error
- `Omega(f)` measures model complexity
- `lambda` controls the accuracy versus simplicity tradeoff

The dashboard must make this tradeoff visible and interactive.

## 2. Dataset requirements

Use the Palmer Penguins dataset.

### 2.1 Dataset source

The project recommends loading the dataset in Python using the `palmerpenguins` package.

Example:

```python
from palmerpenguins import load_penguins
penguins = load_penguins()
```

### 2.2 Dataset features

The dataset contains these eight columns:

- `species`
- `island`
- `sex`
- `year`
- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

Depending on the package version, column names may contain underscores instead of spaces. Keep the meaning consistent with the project sheet.

### 2.3 Target variable

The target feature is:

```text
species
```

It has three categorical classes:

- Adelie
- Gentoo
- Chinstrap

### 2.4 Input features

All columns except `species` are input features:

- `island`
- `sex`
- `year`
- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

### 2.5 Numerical features for PDP and ALE

For Task 5, the user must be able to select one of these four numerical features:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

Each PDP and ALE plot must show the effect of the selected feature on the predicted probability of each species.

## 3. Complete project requirements

## 3.1 Global interface requirements

The final result should be one connected interactive dashboard.

The interface must allow the user to select:

- model class: decision tree or logistic regression
- sparsity or regularization tradeoff using a `lambda` slider
- an example `x` from the dataset for counterfactual explanations
- a target species for counterfactual explanations
- one numerical feature for PDP and ALE plots

The interface must update consistently when the selected model class or `lambda` changes.

This means the following parts must all depend on the same currently selected model:

- displayed model
- test accuracy
- complexity value
- counterfactual explanations
- PDP plot
- ALE plot

## 3.2 Task 1 requirements: Decision tree model

### Required implementation

Fit a decision tree classifier on the Palmer Penguins dataset.

### Required interface output

The user interface must show:

- the resulting decision tree
- the test accuracy
- the number of leaves

### Required model complexity

For a decision tree:

```text
Omega(f) = number of leaves
```

### Recommended scikit-learn model

```python
from sklearn.tree import DecisionTreeClassifier
```

### Recommended visualization options

Use one of these:

```python
from sklearn.tree import plot_tree
```

or:

```python
from sklearn.tree import export_text
```

or:

```python
from sklearn.tree import export_graphviz
```

### Required accuracy metric

Use test accuracy:

```python
accuracy_score(y_test, y_pred)
```

### Required leaf count

Use:

```python
model.get_n_leaves()
```

## 3.3 Task 2 requirements: Decision tree regularization and lambda slider

### Required implementation

Train several decision tree models with varying degrees of regularization.

For scikit-learn decision trees, use:

```python
DecisionTreeClassifier(max_leaf_nodes=...)
```

Example grid:

```python
max_leaf_nodes_grid = [2, 3, 4, 5, 6, 8, 10, 15, 20, None]
```

### Important distinction

There are two different regularization related quantities:

1. The model fitting regularization parameter, such as `max_leaf_nodes`.
2. The interface slider `lambda`, which is used after training to select the best model according to the project objective.

The project explicitly says that `lambda` is different from the parameter used to control regularization during model fitting.

### Required model selection logic

**(Updated per the official project sheet.)** For every trained model `f`, compute:

```text
model_selection_score = acc_test - lambda * Omega(f)
```

where:

- `acc_test` is the model test accuracy
- `Omega(f)` is the complexity (number of leaves for a tree)
- `lambda` comes from the interface slider

The interface must display the model corresponding to the **maximizer** of this quantity.

### Note

The official sheet uses `acc_test - lambda * Omega(f)`, maximized. This is the
mathematically consistent accuracy/complexity tradeoff: higher accuracy raises the
score, more complexity lowers it, and `lambda` controls the penalty.

It is exactly equivalent to **minimizing** `(1 - acc_test) + lambda * Omega(f)`
(the two differ only by the additive constant 1), so either form selects the same
model. The implementation uses the sheet's form directly (`acc_test - lambda*Omega`,
maximized) in `services/selection.py`.

## 3.4 Task 3 requirements: Logistic regression with model complexity

### Required implementation

Repeat the same regularization and lambda based model selection process using logistic regression.

This means you must:

- train several logistic regression models with different regularization strengths
- compute test accuracy for each model
- compute a complexity measure `Omega(f)` for each model
- use the same `lambda` slider concept to select the model
- show the selected model information in the interface

### Recommended model

Use:

```python
from sklearn.linear_model import LogisticRegression
```

For multiclass classification, use multinomial logistic regression.

Example:

```python
LogisticRegression(
    penalty="l1",
    solver="saga",
    multi_class="multinomial",
    C=C_value,
    max_iter=5000
)
```

### Recommended complexity measure

A suitable complexity measure is:

```text
Omega(f) = number of nonzero coefficients
```

This is recommended because it measures sparsity. A logistic regression model that uses fewer nonzero features is easier to inspect.

Alternative valid choices:

```text
Omega(f) = L1 norm of coefficients
```

or:

```text
Omega(f) = L2 norm of coefficients
```

The clearest choice for this project is the number of nonzero coefficients, especially if you use L1 regularization.

### Important scikit-learn detail

For logistic regression:

```text
small C = stronger regularization
large C = weaker regularization
```

This is because `C` is the inverse of regularization strength.

Example grid:

```python
C_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
```

### Required preprocessing

Logistic regression requires preprocessing:

- one-hot encode categorical features
- scale numerical features
- handle missing values

Use a scikit-learn pipeline with `ColumnTransformer`.

Example structure:

```python
numeric_features = [
    "year",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

categorical_features = ["island", "sex"]
```

For Task 5, only use the four biometric numerical features for PDP and ALE selection.

## 3.5 Task 4 requirements: Counterfactual explanations

### Required interface region

Add a region named something like:

```text
Counterfactual explanations
```

### Required user controls

The user must be able to select:

- an example `x` from the dataset
- a target species label
- model class: decision tree or logistic regression
- `lambda` value or sparsity setting

The counterfactuals must be generated using the currently selected model.

### Required method

Implement counterfactual generation as described in the project:

1. Randomly sample `N` points locally around the selected example `x`.
2. Predict each sampled point using the currently selected model.
3. Keep only sampled points whose predicted class is the desired target class.
4. Rank valid points by MAD weighted L1 distance to `x`.
5. Display the best `k` counterfactuals to the user.

### Required distance

Use MAD weighted L1 distance for numerical features:

```text
distance_numeric(x, z) = sum_j |x_j - z_j| / MAD_j
```

where:

```text
MAD_j = median_i |x_ij - median_i(x_ij)|
```

Use a small epsilon to avoid division by zero:

```text
MAD_j = max(MAD_j, epsilon)
```

### Required handling of feature types

The project explicitly asks you to handle decimal, binary, and categorical data correctly.

#### Decimal numerical features

Use local Gaussian noise:

```text
z_j = x_j + Normal(0, sigma_j)
```

Recommended:

```text
sigma_j = alpha * standard_deviation_j
```

Clip values to the valid observed range of the feature.

#### Integer or discrete numerical features

For `year`, treat it as discrete:

- sample from valid observed years
- or add small integer noise and round
- keep values within the observed years

#### Binary features

If a binary feature exists after preprocessing, flip it with a small probability:

```text
P(flip) = p
```

#### Categorical features

For features such as `island` and `sex`, do not add Gaussian noise.

Use one of these approaches:

1. Keep the original category with high probability and switch to another valid category with low probability.
2. Sample from the empirical distribution of categories.
3. Sample uniformly from valid categories.

Recommended local categorical perturbation:

```text
with probability 0.8: keep original category
with probability 0.2: sample another valid category
```

### Required fallback behavior

If no counterfactuals are found:

- increase `N`
- increase the sampling variance for numerical features
- increase the probability of categorical changes
- repeat iteratively until counterfactuals are found or a maximum number of attempts is reached

### Required output

Display the best `k` counterfactuals in a table.

The table should include:

- changed feature values
- predicted class
- predicted probability of target class
- MAD weighted distance
- which features changed

Optional but useful:

- original example values
- difference from original example
- number of changed features

## 3.6 Task 5 requirements: PDP and ALE plots

### Required interface region

Add a region named something like:

```text
Feature effect plots
```

### Required user controls

The user must be able to select:

- model class: decision tree or logistic regression
- `lambda` value or sparsity setting
- one of the four numerical features

The plots must use the currently selected model.

### Required selected features

The selectable features must be:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

### Required plots

For the selected numerical feature, show both:

- PDP plot
- ALE plot

### Required curves per plot

Each plot must contain three curves:

- predicted probability of Adelie
- predicted probability of Gentoo
- predicted probability of Chinstrap

### Required implementation constraint

You must write the code for PDP and ALE yourself.

Do not use a library function that directly computes PDP or ALE values for you.

Allowed:

- use the model's `predict_proba`
- use `numpy` and `pandas` for computations
- use `matplotlib`, `plotly`, or similar for plotting

Not allowed:

- library functions that directly compute PDP
- library functions that directly compute ALE

## 4. PDP implementation knowledge

A partial dependence plot shows the average model prediction when the selected feature is fixed to different values.

For selected feature `j`:

1. Choose a grid of values for feature `j`.
2. For each grid value `v`:
   - copy the dataset
   - replace feature `j` with `v` for every row
   - compute predicted probabilities using the selected model
   - average predicted probabilities over all rows
3. Plot grid values against average predicted probability.

Pseudocode:

```python
def compute_pdp(model, X, feature, grid):
    pdp_values = []
    for v in grid:
        X_temp = X.copy()
        X_temp[feature] = v
        probs = model.predict_proba(X_temp)
        pdp_values.append(probs.mean(axis=0))
    return np.array(pdp_values)
```

Output shape:

```text
number_of_grid_points x number_of_classes
```

For this project:

```text
number_of_classes = 3
```

## 5. ALE implementation knowledge

Accumulated local effects show how a feature locally changes model predictions while reducing unrealistic extrapolation compared with PDP.

For selected feature `j`:

1. Divide the observed values of feature `j` into bins.
2. For each bin `[lower, upper]`:
   - find rows whose feature value lies in that bin
   - create two copies of those rows
   - set feature `j` to `lower` in one copy
   - set feature `j` to `upper` in the other copy
   - compute predicted probabilities for both copies
   - calculate the average difference in predicted probabilities
3. Accumulate the average differences across bins.
4. Center the accumulated effects so the average effect is zero.
5. Plot bin centers against accumulated local effects.

Pseudocode:

```python
def compute_ale(model, X, feature, bins):
    effects = []
    centers = []

    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (X[feature] >= lower) & (X[feature] <= upper)
        X_bin = X.loc[mask].copy()

        if len(X_bin) == 0:
            effects.append(np.zeros(n_classes))
            centers.append((lower + upper) / 2)
            continue

        X_low = X_bin.copy()
        X_high = X_bin.copy()
        X_low[feature] = lower
        X_high[feature] = upper

        diff = model.predict_proba(X_high) - model.predict_proba(X_low)
        effects.append(diff.mean(axis=0))
        centers.append((lower + upper) / 2)

    ale = np.cumsum(np.array(effects), axis=0)
    ale = ale - ale.mean(axis=0)
    return np.array(centers), ale
```

## 6. ALE partial derivative question

The project asks:

```text
For ALE, you will need partial derivatives. For which model can you compute them exactly? For which do you have to use a discretization?
```

### Logistic regression

For logistic regression, partial derivatives can be computed exactly because the model is differentiable.

For multinomial logistic regression:

```text
p_c(x) = exp(beta_c^T x + b_c) / sum_r exp(beta_r^T x + b_r)
```

The derivative of class probability `p_c` with respect to feature `x_j` is:

```text
d p_c / d x_j = p_c * (beta_cj - sum_r p_r * beta_rj)
```

If the feature is standardized, adjust the derivative by the scaling factor.

### Decision tree

For decision trees, exact useful partial derivatives are not available.

Reason:

- a decision tree is piecewise constant inside leaves
- predictions jump at split thresholds
- the derivative is zero almost everywhere and undefined at thresholds

Therefore, for a decision tree you should use discretization or finite differences for ALE.

The finite difference ALE implementation described above works for both decision trees and logistic regression.

## 7. Required preprocessing knowledge

## 7.1 Missing values

The Palmer Penguins dataset contains missing values. You must handle them before training.

Valid approaches:

- drop rows with missing values
- impute numerical features with median
- impute categorical features with most frequent value

Recommended for simplicity:

```python
penguins = penguins.dropna()
```

Recommended for a more robust pipeline:

```python
SimpleImputer(strategy="median")
SimpleImputer(strategy="most_frequent")
```

## 7.2 Train test split

Use a train test split so `test accuracy` is meaningful.

Recommended:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Use `stratify=y` because the target has three classes.

## 7.3 Encoding categorical features

Categorical features:

- `island`
- `sex`

Use one-hot encoding:

```python
OneHotEncoder(handle_unknown="ignore")
```

## 7.4 Scaling numerical features

Logistic regression should use scaled numerical features:

```python
StandardScaler()
```

Decision trees do not require scaling, but using a consistent preprocessing pipeline is acceptable.

## 8. Required machine learning knowledge

You need to understand:

- supervised classification
- multiclass classification
- train test split
- model fitting
- model prediction
- class probabilities using `predict_proba`
- accuracy evaluation
- decision trees
- logistic regression
- regularization
- sparsity
- model complexity measures
- preprocessing pipelines
- one-hot encoding
- feature scaling
- missing value handling

## 9. Required explainability knowledge

You need to understand:

- interpretability versus explainability
- interpretable models
- decision tree interpretability
- logistic regression coefficient interpretation
- complexity penalty `Omega(f)`
- accuracy complexity tradeoff
- counterfactual explanations
- local perturbation sampling
- MAD weighted L1 distance
- categorical perturbations
- binary perturbations
- PDP
- ALE
- partial derivatives for differentiable models
- finite difference approximation for non-differentiable models

## 10. Required programming knowledge

You need to know these Python libraries or equivalents:

### Data processing

```python
pandas
numpy
palmerpenguins
```

### Machine learning

```python
scikit-learn
```

Important scikit-learn components:

```python
DecisionTreeClassifier
LogisticRegression
train_test_split
accuracy_score
ColumnTransformer
Pipeline
OneHotEncoder
StandardScaler
SimpleImputer
```

### Visualization

Use one or more:

```python
matplotlib
plotly
graphviz
sklearn.tree.plot_tree
```

### Interface

Use one of:

```python
streamlit
Dash
Gradio
```

Recommended for this project:

```python
streamlit
```

because it is fast to build dashboards with sliders, dropdowns, tables, and plots.

## 11. Recommended dashboard structure

A good dashboard can be organized like this:

## 11.1 Sidebar controls

- model type selector: `Decision Tree` or `Logistic Regression`
- lambda slider
- optional random seed
- optional number of sampled counterfactual candidates `N`
- optional number of returned counterfactuals `k`

## 11.2 Main area: selected model

Show:

- selected model type
- selected fitting regularization parameter, such as `max_leaf_nodes` or `C`
- selected lambda value
- test accuracy
- complexity `Omega(f)`

For decision tree:

- show tree plot or tree text
- show number of leaves

For logistic regression:

- show coefficient table
- show number of nonzero coefficients

## 11.3 Counterfactual region

Controls:

- selected row or example index
- target species
- number of candidates `N`
- number of counterfactuals `k`

Outputs:

- original prediction
- target class
- table of best counterfactuals
- MAD weighted distance
- changed features
- predicted probabilities

## 11.4 Feature effect plots region

Controls:

- feature selector among the four numerical features
- number of grid points for PDP
- number of bins for ALE

Outputs:

- PDP plot with three curves
- ALE plot with three curves

## 12. Recommended implementation order

Follow this order:

1. Load Palmer Penguins data.
2. Clean missing values.
3. Split into train and test data.
4. Define preprocessing for numerical and categorical features.
5. Train a grid of decision trees.
6. Compute accuracy and number of leaves for each tree.
7. Train a grid of logistic regression models.
8. Compute accuracy and complexity for each logistic model.
9. Implement lambda based model selection.
10. Build the first version of the dashboard with model selection and metrics.
11. Add decision tree visualization.
12. Add logistic regression coefficient display.
13. Implement counterfactual sampling for numerical features.
14. Extend counterfactual sampling to categorical and binary features.
15. Add MAD weighted distance ranking.
16. Add counterfactual table to the dashboard.
17. Implement PDP manually.
18. Implement ALE manually.
19. Add PDP and ALE plots to the dashboard.
20. Test that all dashboard outputs update when model type or lambda changes.
21. Write a short explanation of your design choices.

## 13. Model complexity choices

## 13.1 Decision tree complexity

Required:

```text
Omega(tree) = number of leaves
```

Implementation:

```python
complexity = tree_model.get_n_leaves()
```

## 13.2 Logistic regression complexity

Recommended:

```text
Omega(logistic regression) = number of nonzero coefficients
```

Implementation idea:

```python
coefs = logistic_model.coef_
complexity = np.sum(np.abs(coefs) > threshold)
```

Use a small threshold because coefficients may be numerically close to zero:

```python
threshold = 1e-6
```

Alternative:

```python
complexity = np.sum(np.abs(coefs))
```

or:

```python
complexity = np.linalg.norm(coefs)
```

Choose one and clearly state it.

## 14. Accuracy and selection checklist

For every trained model store:

- model type
- fitting regularization parameter
- test accuracy
- complexity value
- trained pipeline or model object

Then for the selected `lambda`, compute the official score and select its
**maximizer**:

```python
score = test_accuracy - lambda_value * complexity
best_model = model_with_highest_score
```

(Equivalently you may minimize `(1 - test_accuracy) + lambda_value * complexity`;
both select the same model.)

## 15. Counterfactual implementation checklist

Your counterfactual code should:

- accept the selected model
- accept an original example `x`
- accept a target class
- sample candidate points near `x`
- handle numerical features using noise
- handle categorical features using valid category sampling
- handle binary features using flips
- keep candidates predicted as the target class
- compute MAD weighted L1 distance
- rank candidates by distance
- return the best `k` candidates
- increase search radius or `N` if no candidates are found
- display the results in the interface

## 16. PDP implementation checklist

Your PDP code should:

- accept the selected model
- accept dataset `X`
- accept selected numerical feature
- create a grid over observed feature values
- replace the selected feature with each grid value
- call `predict_proba`
- average probabilities over all rows
- return one curve per class
- plot three curves
- not use a PDP library function

## 17. ALE implementation checklist

Your ALE code should:

- accept the selected model
- accept dataset `X`
- accept selected numerical feature
- create bins using observed feature values
- compute local finite differences inside each bin
- average differences within each bin
- accumulate effects across bins
- center the curves
- return one curve per class
- plot three curves
- not use an ALE library function

## 18. Final deliverables checklist

Your final project should include:

- working Python code
- interactive dashboard
- trained decision tree models
- trained logistic regression models
- lambda slider
- model class selector
- model accuracy display
- model complexity display
- decision tree visualization
- logistic regression complexity display
- counterfactual explanation region
- valid counterfactual sampling for numerical, categorical, and binary data
- MAD weighted L1 ranking
- best `k` counterfactual display
- feature effect plots region
- numerical feature selector
- manually implemented PDP
- manually implemented ALE
- three class probability curves per PDP plot
- three class probability curves per ALE plot
- explanation of which model has exact ALE derivatives
- explanation that decision trees need discretization or finite differences for ALE

## 19. Common mistakes to avoid

- Do not train only one model for Task 2. You need several models with different regularization levels.
- Do not confuse the UI `lambda` with `max_leaf_nodes` or `C`.
- Do not use the target column `species` as an input feature.
- Do not forget to handle missing values.
- Do not use Gaussian noise for categorical variables.
- Do not generate counterfactuals from a different model than the one selected in the interface.
- Do not compute PDP or ALE with a ready-made library function.
- Do not make PDP and ALE show only one class. Each plot must show three curves.
- Do not let the lambda slider affect only the displayed metrics. It must affect the selected model used for counterfactuals, PDP, and ALE.
- Do not ignore the exact versus approximate derivative question for ALE.
- Do not present `year` as one of the four numerical features for Task 5 unless the instructor explicitly allows it. The project asks for four numerical features, which are the four biometric measurements.

## 20. Suggested report explanation

In the written explanation or comments, include:

1. Which preprocessing was used.
2. Which decision tree regularization grid was used.
3. Which logistic regression regularization grid was used.
4. Which complexity measure was used for logistic regression and why.
5. How lambda selects among already trained models.
6. How counterfactual candidates are sampled.
7. How categorical and binary features are perturbed.
8. How MAD weighted L1 distance is computed.
9. How PDP is computed manually.
10. How ALE is computed manually.
11. Why logistic regression has exact partial derivatives.
12. Why decision trees require discretization or finite differences.
13. The selection formula `acc_test - lambda * Omega(f)` (maximized), and any notes about it.

## 21. Minimal acceptance criteria

A minimal complete submission must satisfy all of the following:

- It trains a decision tree on Palmer Penguins.
- It shows the decision tree, test accuracy, and number of leaves.
- It trains multiple decision trees with different regularization levels.
- It has a lambda slider for selecting among the trained decision trees.
- It repeats the regularization and complexity logic for logistic regression.
- It defines and reports a logistic regression complexity measure.
- It has a counterfactual region.
- It lets the user select an example and target class.
- It generates counterfactuals by local random sampling.
- It ranks counterfactuals by MAD weighted L1 distance.
- It handles numerical, categorical, and binary feature perturbations appropriately.
- It links counterfactuals to the selected model type and lambda.
- It has a feature effect plots region.
- It lets the user select one of the four numerical biometric features.
- It shows a PDP plot with three species probability curves.
- It shows an ALE plot with three species probability curves.
- It implements PDP and ALE manually.
- It explains exact derivatives for logistic regression and discretization for decision trees.
