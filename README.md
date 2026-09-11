# Human-Centric Artificial Intelligence: Course Projects

This repository holds our four projects for the Human-Centric Artificial
Intelligence course. As the course asks, they all live in one Django project,
with one app per project, and you reach each of them from a shared home page.

Ehab Mansour, Elyes Oueslati

## Running it

We developed and tested everything with Python 3.12, so that is the safest
version to use. Django 5 needs at least Python 3.10.

Clone the repository, create a virtual environment and install the
requirements:

```bash
git clone https://github.com/IhabRagabMansour/HCAI-Project01.git
cd HCAI-Project01
python -m venv venv
```

Activate it. On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Then install and start the server:

```bash
pip install -r requirements.txt
python manage.py runserver
```

Open <http://127.0.0.1:8000/> and pick a project from the home page.

That is all it takes. The database ships already migrated and empty, so you
don't have to run `migrate` first. If you ever delete `db.sqlite3` or want a
fresh one, `python manage.py migrate` rebuilds it.

Nothing needs an internet connection once the packages are installed. Every
dataset is in the repository, and so is Chart.js, which draws the charts.

## The projects

### Project 1: Automated Machine Learning

A small interface for supervised learning. You upload a CSV file (the last
column is taken as the target), look through the data, and train models on it.

On the data side you can page and sort through the whole table, plot
histograms and scatter plots, and check whether the target classes are
balanced. The app decides whether the task is classification or regression
from the target column, and you can override that when it guesses wrong, for
example when classes are stored as numbers.

Training happens in experiments. An experiment fixes the preprocessing
(missing values, categorical encoding, outliers, scaling, oversampling for
imbalanced classes) and the train/test split. Inside it you can train several
models: logistic or linear regression, random forest, SVM, k-nearest neighbours
and decision trees, with the hyperparameters either set by hand or tuned by a
randomised search with cross-validation. Each trained model gets a results page with test metrics,
cross-validation scores, a confusion matrix, a learning curve and feature
importances. You can also predict a single new row, compare models across
experiments, and download a trained model.

`data/iris.csv` is a small file to try it with.

### Project 2: Explainability

An explainability dashboard for models trained on the Palmer Penguins dataset.
A decision tree and an L1-regularised logistic regression are each trained at
several levels of regularisation, and a lambda slider picks the model that
maximises `acc_test - lambda * Omega(f)`, where Omega is the number of leaves
for the tree and the number of nonzero coefficients for the logistic
regression.

The rest of the dashboard follows whichever model is selected. You can pick an
example and a target species and get counterfactual explanations, ranked by
MAD-weighted L1 distance, and you can look at partial dependence and
accumulated local effects plots for the four body measurements. We wrote the
PDP and ALE code ourselves rather than using a library, as the project sheet
asks.

A report explaining our choices can be read in the app or downloaded as a PDF.

### Project 3: Active Learning for Learning-to-Defer

A news-topic classifier on AG News that can hand an article to a simulated
expert instead of answering itself. The expert is very good on Business and
Sci/Tech, the two topics the classifier mixes up, and weak everywhere else.

The app walks through the four tasks: the baseline classifier, the simulated
expert and where it is strong, a learning-to-defer rule that decides when to
defer, and active learning to find out where the expert is competent while
asking it as few questions as possible. There is also a page where you play
the expert yourself. The PDF report on the landing page explains and
justifies each decision.

The first time you open one of the Project 3 pages it trains the classifier on
120,000 articles, which takes about 40 seconds. The results are cached in
`project3/artifacts/` afterwards, so later visits are instant. Delete that
folder if you want to retrain from scratch.

### Project 4: Preference Elicitation

A user study comparing two ways of learning a new user's taste in films:
choosing between two films at a time, or ranking ten. The landing page has the
PDF report, which describes the full study design, and a button to start the
study itself.

The study is the interface a participant would actually see. It runs through
consent, a short background questionnaire, instructions and practice, the two
elicitation methods in counterbalanced order with a questionnaire after each,
a set of held-out choices used to compare the two fitted models, and a
debrief. Responses are stored under a random participant code, and the
landing page can export them as CSV for analysis.

## Running the tests

```bash
python manage.py test
```

There are 767 tests across the four projects. The full run takes a few
minutes, mostly because the Project 3 tests train on the full dataset.

## Where things are

```
pbl/          Django settings and the root URL configuration
home/         the home page listing the projects
project1/     Automated Machine Learning
project2/     Explainability
project3/     Active Learning for Learning-to-Defer
project4/     Preference Elicitation
demos/        the demo app that came with the course skeleton
data/         iris.csv, movie_metadata.csv (IMDB 5000) and agnews.csv.gz
templates/    the shared base template
static/       the shared stylesheet and favicon
media/        uploaded CSV files and trained models from Project 1
```

Each project app keeps its logic in a `services/` folder, separate from the
Django views, so the machine learning code can be tested on its own. The PDF
reports are generated when you download them, from the same code and data the
app uses, so their numbers always match what the app shows.

## Dependencies

Django, NumPy, pandas, scikit-learn and matplotlib, plus three smaller
packages: `imbalanced-learn` for oversampling in Project 1, `palmerpenguins`
for the Project 2 dataset (the project sheet suggests it), and `reportlab` for
the PDF reports. The exact versions are in `requirements.txt`.
