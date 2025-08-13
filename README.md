## Diabetes ML Pipeline with Streamlit

End-to-end binary classification project for diabetes prediction using Streamlit and scikit-learn. Includes data exploration, interactive visualizations, real-time predictions, and a comprehensive model performance dashboard (metrics, confusion matrix, ROC/PR curves, and model comparisons).

### Key Features
- **Data exploration**: Browse shape, columns, head, and interactively filter ranges.
- **Visualizations**: Plotly histograms, scatter plots with optional trendline, and a correlation heatmap.
- **Model prediction (clinical-first)**: ADA rules applied first (FPG/OGTT/HbA1c/random+symptoms). If diagnostic criteria are met, classification is clinically overridden; otherwise the ML model acts as a risk stratifier with bands (Low/Moderate/High/Very High). Unified output shows Clinical Category, Decision Source, Model Risk Probability (+ band), and Next Step.
- **Model performance**:
  - **Metrics**: ROC-AUC, Brier score, Accuracy, Precision, Recall, F1.
  - **Charts**: Confusion matrix heatmap, ROC curve, Precision–Recall curve.
  - **Model comparison**: Per-model holdout results (sorted by ROC-AUC with Brier tie-break).
- **Reproducible split**: Holdout uses a fixed `random_state=42` and stratification to match the training notebook.

## Project Structure
- `app.py`: Streamlit app with navigation, EDA, visualizations, prediction, and performance pages.
- `model.pkl`: Trained model pipeline with embedded metrics produced by the training notebook.
- `data/dataset.csv`: Pima Indians Diabetes dataset (binary target `Outcome`).
- `requirements.txt`: Python dependencies (Python 3.11 recommended).
- `runtime.txt`: Optional; set interpreter version for hosted deployments.
- `notebooks/model_training.ipynb`: EDA + training notebook that evaluates multiple models and saves `model.pkl` with metrics.
- `rule_wrappers.py`: Optional clinical override wrapper (see below).

## Folder Structure
```text
Diabetes/
├─ app.py                      # Streamlit app entrypoint
├─ data/
│  └─ dataset.csv              # Input dataset (Pima Indians Diabetes)
├─ model.pkl                   # Saved best pipeline + metrics (created by notebook)
├─ notebooks/
│  └─ model_training.ipynb     # EDA + training; writes model.pkl with metrics
├─ README.md                   # Project documentation
├─ requirements.txt            # Python dependencies
├─ rule_wrappers.py            # Optional clinical rule wrapper for predictions
├─ runtime.txt                 # Optional Python runtime pin for deployment
└─ venv/                       # (Local) Virtual environment, not required in repo
```

## Setup
### 1) Python and virtual environment
- **Python**: 3.11 is recommended.

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Dataset
- Ensure `data/dataset.csv` exists. The app expects columns:
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`
  - Optional: `measurement_type` with values such as `fasting`, `post_meal`, `random`, `ogtt_2h` (defaulted to `fasting` if absent). For `ogtt_2h`, the `Glucose` value is interpreted as the 2-hour value.
  - Optional (advanced): If you collect lab values separately, you may map them to the decision helper as `hba1c` and `ogtt_2h`.

## Training
Training is performed via the notebook `notebooks/model_training.ipynb`:
- Performs preprocessing (imputation, scaling, one-hot for categorical) and evaluates at least Logistic Regression and Random Forest.
- Uses 5-fold CV (ROC-AUC) and a stratified 80/20 holdout split.
- Selects the best model by ROC-AUC (breaking ties by lower Brier score).
- Saves a dictionary to `model.pkl` containing:
  - `pipeline`: the best fitted pipeline.
  - `metrics`: `{ best_model, holdout_results, feature_names, zero_as_missing_columns }`.

Note: The app can fall back to `training_metrics.json` if present, but `model.pkl` is the preferred single artifact.

## Run the App
```bash
streamlit run app.py
```
Navigate to the pages via the left sidebar:
- Overview
- Data Exploration
- Visualizations
- Model Prediction
- Model Performance

## Model Performance Page
- **Top section** shows the selected best model name and a sortable per-model table from the saved `holdout_results`.
- **Detailed evaluation** recomputes metrics on the notebook’s holdout split (same `random_state=42`) using the saved best pipeline.
- Displays:
  - Metrics: ROC-AUC, Brier, Accuracy, Precision, Recall, F1.
  - Confusion matrix heatmap.
  - ROC and Precision–Recall curves.

## Clinical-first Decision Logic (Unified Output)
The app applies ADA-style clinical thresholds before the ML prediction:
- Fasting Plasma Glucose (FPG): <100 Normal; 100–125 Prediabetes; ≥126 Diabetes (confirm if asymptomatic)
- 2h OGTT: <140 Normal; 140–199 Prediabetes; ≥200 Diabetes
- HbA1c: <5.7 Normal; 5.7–6.4 Prediabetes; ≥6.5 Diabetes (confirm if asymptomatic)
- Random glucose ≥200 mg/dL + symptoms → Diabetes (single test sufficient)

If any diagnostic rule is met, the app marks the case accordingly (Decision Source: "Clinical override" or "Rule + ML"). Otherwise, the ML model decides (Decision Source: "ML only"). The UI always shows:
- **Clinical Category**: Normal, Prediabetes, or Diabetes
- **Decision Source**: Clinical override, Rule + ML, or ML only
- **Model Risk Probability**: Percentage plus risk band
  - Bands: <20% Low; 20–50% Moderate; 50–80% High; ≥80% Very High
- **Next Step** guidance

The training notebook includes the same helper functions, ensuring consistent decisions in both environments.

## Optional Clinical Wrapper (legacy)
`rule_wrappers.py` contains a simple glucose-threshold override wrapper. The primary app now ships with a built-in, more complete clinical-first decision layer; the wrapper remains for reference or specialized use.

## Deployment
- **Streamlit Community Cloud**
  - Include `requirements.txt` and add `runtime.txt` with a single line `3.11`.
  - Set the app entrypoint to `app.py`.
- **Heroku (or similar)**
  - Include `requirements.txt` and a `runtime.txt` like `python-3.11.9`.
  - Use a Procfile if required by the platform: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Troubleshooting
- **model.pkl not found**: Run the training notebook to generate `model.pkl` in the project root, then restart the app.
- **Missing dataset**: Ensure `data/dataset.csv` exists and has the required columns.
- **Charts not rendering**: Verify Plotly/Matplotlib/Seaborn are installed (`pip install -r requirements.txt`).
- **Statsmodels trendline warning**: Install `statsmodels` to enable scatter plot trendlines.
- **CORS/XSRF issues on hosted envs**: Consider Streamlit flags `--server.enableXsrfProtection=false --server.enableCORS=false` (use with caution).

## Repository Links
- Main repo: `https://github.com/DevAyesh/streamlit-diabetes-predictor` ([source](https://github.com/DevAyesh/streamlit-diabetes-predictor))

## License
No explicit license provided. If you plan to release this publicly, consider adding a license (e.g., MIT, Apache-2.0).

## Acknowledgements
- Dataset: Pima Indians Diabetes dataset.
- Built with: Python, scikit-learn, Streamlit, Plotly, Seaborn, Matplotlib, pandas, numpy.

