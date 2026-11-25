# Traffic Prediction

## Project Overview

This project uses machine-learning regression methods to predict traffic flow on westbound Interstate 8. The project uses timestamped traffic observations recorded at five-minute intervals. Historical traffic values and time-based features are used to predict traffic flow for future observations.

The project compares linear regression, gradient boosting regression, random forest regression, and K-nearest neighbors regression. The models are evaluated using mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination (R²).

## AI Assistance Disclosure

This README file was created with assistance from ChatGPT. ChatGPT was also used to help organize the repository, explain machine-learning concepts, generate initial repo structure, debug errors, and suggest documentation. All team members are responsible for reviewing, testing, understanding, and modifying the code before submitting the project.

## Repository Structure

```text
Traffic_Prediction/
|-- data/
|   |-- i8_west_2021.csv
|-- prediction_model/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- evaluation.py
|   |-- train.py
|   |-- models/
|       |-- __init__.py
|       |-- linear_regression.py
|       |-- gradient_boosting.py
|       |-- random_forest.py
|       |-- knn.py
|-- saved_models/
|-- results/
|-- app/
|-- requirements.txt
|-- README.md
```

## Installation

Use Python 3.10 or newer if possible. From the repository root, install dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

Streamlit will be needed when the application is added:

```bash
pip install streamlit
```

## Dataset

Place the dataset at:

```text
data/i8_west_2021.csv
```

The current dataset contains two original columns:

- `timestamp`: date and time of the traffic observation
- `flow`: measured traffic flow and prediction target

The preprocessing code creates additional features from these columns, including hour, minute, day of week, month, weekend status, previous traffic flow, and rolling traffic averages.

## Running the Models

Run commands from the repository root:

```bash
python prediction_model/train.py data/i8_west_2021.csv --model linear_regression
python prediction_model/train.py data/i8_west_2021.csv --model gradient_boosting
python prediction_model/train.py data/i8_west_2021.csv --model random_forest
python prediction_model/train.py data/i8_west_2021.csv --model knn
```

The data is split chronologically. Earlier observations are used for training, and later observations are used for testing. This prevents future traffic observations from leaking into training.

## Generated Outputs

After training, the repository creates a saved model and report-ready results:

```text
saved_models/
|-- linear_regression.joblib

results/
|-- linear_regression/
|   |-- metrics.json
|   |-- predictions.csv
|   |-- coefficients.csv
|-- gradient_boosting/
|   |-- metrics.json
|   |-- predictions.csv
|-- random_forest/
|   |-- metrics.json
|   |-- predictions.csv
|-- knn/
    |-- metrics.json
    |-- predictions.csv
```

`metrics.json` contains MAE, RMSE, and R². `predictions.csv` contains timestamps, actual flow, predicted flow, and prediction error. `coefficients.csv` is created for linear regression and shows the relative effect of each feature.

## Evaluation Metrics

- **MAE:** average absolute prediction error. Lower is better.
- **RMSE:** gives additional weight to large errors. Lower is better.
- **R²:** percentage of variation explained by the model. Higher is better.

Every model must use the same features and testing period so the comparison is fair.

### Application

- Build the interactive interface in `app`.
- Load a saved trained model rather than retraining every time.
- Use the same preprocessing steps as training.
- Display a prediction and an explanation or visualization.