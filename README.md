# HCAI Project 1: Automated Machine Learning

This repository contains the implementation of **Project 1** for the course **Human-Centric Artificial Intelligence**.

The goal of this project is to build a small Django web application that allows users to:

- upload a dataset in CSV format,
- visualize the data,
- train supervised machine learning models,
- evaluate model performance through an interactive interface.

## General Project Guidelines

- Projects may be completed individually or in groups of up to **5 students**.
- Larger groups are expected to deliver a more complete and polished solution.
- All course projects must be integrated into **one single Django project**.
- The final submission must be provided as a **Git repository**.
- Group information will be collected later through **StudIP**.

## Repository Structure

The course project is built using **Python** and **Django**. Each project is implemented as a Django app, and all apps are accessible from a shared launch page.

### Existing Home App

The repository already includes a `home` app that acts as the landing page.

Important files in a Django app:

- `models.py`  
  Defines data models. For the home page, this file may remain empty.

- `urls.py`  
  Defines URL routes for the app.

- `views.py`  
  Contains the logic used to generate views.

- `templates/[app_name]/...`  
  Contains the HTML templates used by the app.

### Task 1: Edit the Home Page

Update the home page so that it displays:

- the names of all group members,
- the matriculation numbers of all group members.

This change must be done in **Python code**, not by hardcoding the names directly in the HTML template.

## Running the Project

Clone or download the provided project skeleton:

```bash
python manage.py runserver
```

Then open the application in your browser at:

```text
http://127.0.0.1:8000/
```

## Styling

- A global stylesheet is available in the root `static` directory.
- A project-specific stylesheet for the home page is available in `static/home`.
- Future project apps should follow the same pattern for static files.
- The focus of the course is not UX design, but the interface should remain clean and usable.

## Project 1: Supervised Learning Interface

The first project consists of implementing a supervised learning interface in Django.

### Objective

The app should provide two main functionalities:

1. **Data visualization**
2. **Machine learning model training**

This corresponds to a simple end-to-end supervised learning pipeline.

## Creating the Django App

From the root directory of the Django project, create the app:

```bash
python manage.py startapp project1
```

You may choose another app name if needed, but `project1` is used here as the reference example.

### Required Setup Steps

#### 1. Register the app in `settings.py`

Add the app to `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    ...,
    'project1',
]
```

#### 2. Create `urls.py` inside the app

Create `project1/urls.py` with the following content:

```python
from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
]
```

#### 3. Create an initial view

In `project1/views.py`, add a simple starting view:

```python
from django.http import HttpResponse


def index(request):
    return HttpResponse("Welcome to Project 1!")
```

#### 4. Connect the app to the main project URLs

In the main `urls.py` file of the Django project, include the app routes:

```python
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('home/', include('home.urls')),
    path('admin/', admin.site.urls),
    path('project1/', include('project1.urls')),
]
```

### Task 2: Add the App to the Home Page

Add a link to the new app in the list of projects on the home page:

```python
{"name": "Project 1", "url name": "project1:index"}
```

Adapt the namespace if you choose a different app name.

## Useful Tools

You will likely need functionality for:

- file uploads,
- image display,
- plot generation.

Examples are available in the `demos` app.

### Notes on Plotting

Django does not directly display `matplotlib` figures. A common solution is:

1. generate the figure,
2. save it to the `media` directory,
3. load the saved image into the page.

An alternative is to use a JavaScript visualization library such as **Chart.js**.

## Data Loading and Visualization

The application should work with scalar datasets where:

- the **first CSV row** contains feature names,
- the **last column** contains the target output.

### Example CSV Layout

```csv
SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
5.1,3.5,1.4,0.2,1
4.9,3.0,1.4,0.2,1
```

In this example:

- features are the sepal and petal measurements,
- the target label is the flower species.

If a dataset includes an ID column, you may either:

- assume such datasets are not provided, or
- explicitly filter that column out.

### Supported Problem Types

You may choose to support:

- **classification only**,
- **regression only**, or
- **both classification and regression**.

If both are supported, decide whether:

- the user explicitly selects the problem type, or
- the system detects it automatically.

### Suggested Visualizations

For **classification**:

- scatter plots of two selected features,
- color-coded points by class label.

For **regression**:

- scatter plots of one feature vs target,
- or scatter plots of pairs of features.

### Task 3: Upload, Read, and Visualize Data

Implement a module that allows the user to:

- upload a CSV file,
- parse and read the dataset,
- visualize the data.

The interface design and visualization choices are open, but the minimum expectation is a working solution for these tasks.

## Model Training Pipeline

After loading and visualizing the data, the user should be able to train a machine learning model.

The expected pipeline includes:

- choosing a machine learning model,
- splitting the dataset into training and testing sets,
- training the model for several hyperparameter values,
- evaluating the trained models with a chosen metric.

You may use **scikit-learn** for the machine learning implementation.

### Task 4: Implement the Full Pipeline

Build the complete supervised learning workflow in the app.

Important design questions include:

- Should the user choose the model manually?
- Should train/test splitting be user-controlled or automatic?
- Should the user define hyperparameters, or should defaults be provided?
- Should the evaluation metric be selectable?

If you support multiple learning algorithms, you may need Django models to manage algorithm configurations and related parameters.

## Minimum Expected Features

A solid minimal implementation should include:

- a working Django app integrated into the main project,
- a home page link to the project,
- CSV upload support,
- basic dataset parsing,
- at least one visualization,
- at least one supervised learning model,
- train/test split handling,
- model evaluation display.

## Recommended Enhancements

To improve the project quality, consider adding:

- multiple visualization types,
- support for both classification and regression,
- multiple ML algorithms,
- hyperparameter controls in the UI,
- automatic problem-type detection,
- result comparison tables,
- cleaner layout and styling,
- error handling for malformed CSV files.

## Suggested Tech Stack

- **Python**
- **Django**
- **scikit-learn**
- **matplotlib** or **Chart.js**
- **HTML/CSS**

## Final Deliverable

The final deliverable should be a Git repository containing:

- the shared Django project,
- the `home` app,
- the `project1` app,
- all templates and static files,
- all code required to upload data, visualize it, and train models.

## Summary

This project is about building a human-facing machine learning interface, not just a backend script. The final application should make it easy for users to:

- provide a dataset,
- inspect the data visually,
- train a model,
- understand the output of the training pipeline.

The more complete, robust, and usable the application is, the better the final evaluation is likely to be.
