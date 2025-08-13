Diabetes ML Pipeline with Streamlit

Project Structure
- `app.py`: Streamlit app with navigation, EDA, visualizations, prediction, and performance pages
- `train_model.py`: Trains multiple models, selects best by ROC AUC, saves `model.pkl` and `training_metrics.json`
- `model.pkl`: Trained model pipeline
- `data/dataset.csv`: Dataset
- `requirements.txt`: Dependencies
- `notebooks/model_training.ipynb`: EDA and training launcher

Quickstart
1. Create/activate venv
2. Install deps: `python -m pip install -r requirements.txt`
3. Train: `python train_model.py`
4. Run app: `python -m streamlit run app.py`

Notes
- Handles 0 as missing for select numeric features.
- Metrics saved to `models/training_metrics.json`.

