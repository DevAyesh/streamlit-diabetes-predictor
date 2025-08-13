from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Diabetes ML App", page_icon="🧪", layout="wide")

DATA_PATH = Path("data/dataset.csv")
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("training_metrics.json")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    # Prefer embedded metrics in model.pkl; fallback to training_metrics.json
    if MODEL_PATH.exists():
        try:
            obj = joblib.load(MODEL_PATH)
            if isinstance(obj, dict) and "metrics" in obj:
                return obj["metrics"]
        except Exception:
            pass
    if METRICS_PATH.exists():
        try:
            import json

            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def sidebar_navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        options=[
            "Overview",
            "Data Exploration",
            "Visualizations",
            "Model Prediction",
            "Model Performance",
        ],
    )


def page_overview():
    st.title("Diabetes ML App")
    st.write("A complete ML pipeline with EDA, model training, prediction, and deployment readiness.")
    st.info("Dataset: Pima Indians Diabetes (binary classification)")


def page_data_exploration(df: pd.DataFrame | None):
    st.header("Data Exploration")
    if df is None:
        st.warning("Dataset not found at data/diabetes.csv")
        return
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(20))

    with st.expander("Filter"):
        col = st.selectbox("Column", options=df.columns)
        min_val, max_val = float(df[col].min()), float(df[col].max())
        sel = st.slider("Range", min_val, max_val, (min_val, max_val))
        st.dataframe(df[(df[col] >= sel[0]) & (df[col] <= sel[1])].head(50))


def page_visualizations(df: pd.DataFrame | None):
    st.header("Visualizations")
    if df is None:
        st.warning("Dataset not found at data/diabetes.csv")
        return

    # Plotly histogram
    fig = px.histogram(df, x="Glucose", color="Outcome", barmode="overlay", nbins=40, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    # Pair-like scatter (trendline requires statsmodels)
    try:
        import statsmodels.api as sm  # noqa: F401
        trend = "ols"
    except Exception:
        trend = None
        st.info("Install 'statsmodels' to enable trendline (pip install statsmodels).")

    scatter = px.scatter(df, x="BMI", y="Glucose", color="Outcome", trendline=trend)
    st.plotly_chart(scatter, use_container_width=True)

    # Correlation heatmap
    corr = df.corr(numeric_only=True)
    sns_fig = sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    st.pyplot(sns_fig.get_figure(), clear_figure=True)


def page_model_prediction(pipeline):
    st.header("Model Prediction")
    if pipeline is None:
        st.warning("Model not trained yet. Run training locally and refresh.")
        if st.button("Train model now"):
            with st.spinner("Training model..."):
                import subprocess, sys

                res = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
                st.code(res.stdout or "(no stdout)")
                if res.returncode != 0:
                    st.error("Training failed.")
                    st.code(res.stderr)
                else:
                    st.success("Training complete. Reloading...")
                    st.experimental_rerun()
        return

    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        pregnancies = c1.number_input("Pregnancies", 0, 20, 1)
        glucose = c2.number_input("Fasting Plasma Glucose (mg/dL)", 0.0, 300.0, 120.0)
        blood_pressure = c3.number_input("BloodPressure", 0.0, 200.0, 70.0)
        skin_thickness = c4.number_input("SkinThickness", 0.0, 100.0, 20.0)

        insulin = c1.number_input("Insulin", 0.0, 900.0, 80.0)
        bmi = c2.number_input("BMI", 0.0, 70.0, 28.0)
        dpf = c3.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
        age = c4.number_input("Age", 10, 100, 33)

        mtype = c1.selectbox("Measurement Type", options=["fasting", "post_meal"], index=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        df = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                    "measurement_type": mtype,
                }
            ]
        )
        try:
            prob = float(pipeline.predict_proba(df)[:, 1][0])
            st.metric("Probability of Diabetes", f"{prob:.2%}")
            st.progress(min(1.0, max(0.0, prob)))
            st.write("Predicted:", "Diabetic" if prob >= 0.5 else "Non-diabetic")
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")


def page_model_performance():
    st.header("Model Performance")
    metrics = load_metrics()
    if metrics is None:
        st.info("Metrics not found. Train model first.")
        return
    try:
        # Always show best model if available
        best_model = metrics.get("best_model")
        if best_model:
            st.markdown(f"**Selected best model:** `{best_model}`")

        # New format: list of models with metrics (ROC-AUC, Brier)
        holdout_results = metrics.get("holdout_results")
        if isinstance(holdout_results, list) and holdout_results:
            st.subheader("Per-model holdout metrics")
            try:
                import pandas as pd

                df_res = pd.DataFrame(holdout_results)
                # Ensure consistent columns
                for col in ["roc_auc", "brier"]:
                    if col in df_res.columns:
                        df_res[col] = df_res[col].astype(float)
                # Sort by ROC-AUC desc, then Brier asc
                sort_cols = [c for c in ["roc_auc", "brier"] if c in df_res.columns]
                if sort_cols:
                    df_res = df_res.sort_values(by=["roc_auc", "brier"] if set(sort_cols) == {"roc_auc", "brier"} else sort_cols,
                                                ascending=[False, True] if set(sort_cols) == {"roc_auc", "brier"} else False)
                st.dataframe(df_res.reset_index(drop=True), use_container_width=True)
            except Exception:
                # Fallback simple listing
                for item in holdout_results:
                    st.write(item)
        else:
            # Back-compat: older metrics format
            summary = {k: metrics[k] for k in ["cv_roc_auc_mean", "holdout_roc_auc"] if k in metrics}
            if summary:
                st.json(summary)
            all_models = metrics.get("all_models")
            if isinstance(all_models, dict):
                st.subheader("Per-model metrics")
                for name, m in all_models.items():
                    st.markdown(f"**{name}**")
                    st.write({
                        "cv_roc_auc_mean": m.get("cv_roc_auc_mean"),
                        "cv_roc_auc_std": m.get("cv_roc_auc_std"),
                        "holdout_roc_auc": m.get("holdout_roc_auc"),
                    })
                    if "confusion_matrix" in m:
                        st.write("Confusion Matrix:")
        
        # Compute and display detailed metrics and charts for the best saved pipeline
        st.subheader("Detailed evaluation on holdout set")
        pipeline = load_pipeline()
        df = load_data()
        if pipeline is None or df is None:
            st.info("Model or dataset not available to compute charts.")
            return
        
        if "measurement_type" not in df.columns:
            df["measurement_type"] = "fasting"
        
        feature_names = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        try:
            X = df[feature_names + ["measurement_type"]]
            y = df["Outcome"].astype(int)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to prepare features/labels: {e}")
            return
        
        # Reproduce the training notebook's holdout split
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            roc_auc_score,
            confusion_matrix,
            roc_curve,
            precision_recall_curve,
            accuracy_score,
            precision_recall_fscore_support,
            brier_score_loss,
        )
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        try:
            prob = pipeline.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to predict on holdout set: {e}")
            return
        
        auc = float(roc_auc_score(y_te, prob))
        brier = float(brier_score_loss(y_te, prob))
        acc = float(accuracy_score(y_te, pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="binary", zero_division=0
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC", f"{auc:.3f}")
        c2.metric("Brier score", f"{brier:.3f}")
        c3.metric("Accuracy", f"{acc:.3f}")
        c4.metric("Precision", f"{precision:.3f}")
        c5.metric("Recall", f"{recall:.3f}")
        st.caption(f"F1-score: {f1:.3f}")
        
        # Confusion matrix heatmap
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["Non-diabetic", "Diabetic"], columns=["Pred 0", "Pred 1"])
        st.write("Confusion Matrix")
        ax = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        st.pyplot(ax.get_figure(), clear_figure=True)
        
        # ROC and PR curves
        import plotly.graph_objects as go
        fpr, tpr, _ = roc_curve(y_te, prob)
        prec, rec, _ = precision_recall_curve(y_te, prob)
        
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(roc_fig, use_container_width=True)
        
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
        pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(pr_fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load metrics: {e}")


def main():
    page = sidebar_navigation()
    df = load_data()
    pipeline = load_pipeline()

    if page == "Overview":
        page_overview()
    elif page == "Data Exploration":
        page_data_exploration(df)
    elif page == "Visualizations":
        page_visualizations(df)
    elif page == "Model Prediction":
        page_model_prediction(pipeline)
    elif page == "Model Performance":
        page_model_performance()


if __name__ == "__main__":
    main()


