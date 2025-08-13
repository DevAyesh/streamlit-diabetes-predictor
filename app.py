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


def band_from_prob(probability: float) -> str:
    p = float(probability)
    if p < 0.20:
        return "Low"
    if p < 0.50:
        return "Moderate"
    if p < 0.80:
        return "High"
    return "Very High"


def classify_diabetes(
    inputs: dict,
    model_prob: float,
    threshold: float = 0.5,
    symptomatic: bool = False,
):
    """
    Applies ADA-style clinical rules first; if none met, falls back to ML.

    inputs expected keys (optional where noted):
      - measurement_type: "fasting" | "post_meal" | "random" | "ogtt_2h"
      - glucose: float (mg/dL)
      - ogtt_2h: Optional[float] (mg/dL)
      - hba1c: Optional[float] (percent)
    """
    measurement_type = str(inputs.get("measurement_type") or "").strip().lower()
    glucose = inputs.get("glucose")
    ogtt_2h = inputs.get("ogtt_2h")
    hba1c = inputs.get("hba1c")

    final_category = None  # "Diabetes" | "Prediabetes" | "Normal"
    decision_source = "ML only"
    confirmatory_flag = False

    # 1) Random glucose ≥ 200 mg/dL + symptoms → Diabetes (single test sufficient)
    if measurement_type == "random" and glucose is not None:
        try:
            if float(glucose) >= 200.0 and bool(symptomatic):
                final_category = "Diabetes"
                decision_source = "Clinical override"
        except Exception:
            pass

    # 2) HbA1c thresholds
    if final_category is None and hba1c is not None:
        try:
            a1c = float(hba1c)
            if a1c >= 6.5:
                final_category = "Diabetes"
                decision_source = "Clinical override"
                confirmatory_flag = True  # confirm if asymptomatic
            elif 5.7 <= a1c <= 6.4:
                final_category = "Prediabetes"
                decision_source = "Rule + ML"
            elif a1c < 5.7:
                final_category = "Normal"
                decision_source = "Clinical override"
        except Exception:
            pass

    # 3) OGTT 2h thresholds
    if final_category is None and ogtt_2h is not None:
        try:
            og = float(ogtt_2h)
            if og >= 200.0:
                final_category = "Diabetes"
                decision_source = "Clinical override"
            elif 140.0 <= og <= 199.0:
                final_category = "Prediabetes"
                decision_source = "Rule + ML"
            elif og < 140.0:
                final_category = "Normal"
                decision_source = "Clinical override"
        except Exception:
            pass

    # 4) FPG thresholds (apply when measurement is fasting)
    if final_category is None and measurement_type == "fasting" and glucose is not None:
        try:
            fpg = float(glucose)
            if fpg >= 126.0:
                final_category = "Diabetes"
                decision_source = "Clinical override"
                confirmatory_flag = True  # confirm if asymptomatic
            elif 100.0 <= fpg <= 125.0:
                final_category = "Prediabetes"
                decision_source = "Rule + ML"
            elif fpg < 100.0:
                final_category = "Normal"
                decision_source = "Clinical override"
        except Exception:
            pass

    # Fall back to ML if no clinical diagnosis reached
    prob = float(model_prob)
    if final_category is None:
        final_category = "Diabetes" if prob >= float(threshold) else "Normal"
        decision_source = "ML only"

    # Risk band and next steps
    risk_band = band_from_prob(prob)
    if final_category == "Diabetes":
        next_step = "Discuss confirmatory testing or management with clinician"
    elif final_category == "Prediabetes":
        next_step = "Lifestyle changes + re-test in 3–6 months"
    else:
        next_step = "Maintain healthy lifestyle; consider periodic screening"

    return {
        "clinical_category": final_category,
        "decision_source": decision_source,
        "model_probability": prob,
        "risk_band": risk_band,
        "next_step": next_step,
        "confirmatory_needed": bool(confirmatory_flag),
    }

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
    st.subheader("Predict diabetes risk with an end-to-end ML pipeline")
    st.write(
        "Explore the dataset, visualize patterns, run predictions, and review model performance — all in one place."
    )

    # Quick snapshot metrics
    df = load_data()
    mts = load_metrics()
    num_rows = int(df.shape[0]) if df is not None else None
    num_features = int(df.shape[1]) if df is not None else None
    positive_rate = float(df["Outcome"].mean()) if df is not None and "Outcome" in df.columns else None
    best_model = mts.get("best_model") if isinstance(mts, dict) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{num_rows:,}" if num_rows is not None else "—")
    c2.metric("Columns", f"{num_features}" if num_features is not None else "—")
    c3.metric("Positive rate", f"{positive_rate:.1%}" if positive_rate is not None else "—")
    c4.metric("Best model", best_model if best_model else "—")

    st.markdown("---")

    # How to use
    st.markdown(
        """
        #### Get started
        - Go to **Data Exploration** to inspect the dataset and filter by ranges.
        - Visit **Visualizations** for distributions and relationships.
        - Use **Model Prediction** to input patient features and get risk probability.
        - Check **Model Performance** for metrics, confusion matrix, ROC and PR curves.
        """
    )

    # Mini visual teaser (optional)
    if df is not None:
        try:
            fig = px.histogram(
                df,
                x="Glucose",
                color="Outcome",
                barmode="overlay",
                nbins=35,
                opacity=0.6,
                title="Glucose distribution by outcome (teaser)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # Feature definitions
    with st.expander("Feature definitions"):
        st.markdown(
            """
            - **Pregnancies**: Number of times pregnant
            - **Glucose**: Plasma glucose concentration (mg/dL)
            - **BloodPressure**: Diastolic blood pressure (mm Hg)
            - **SkinThickness**: Triceps skinfold thickness (mm)
            - **Insulin**: 2-Hour serum insulin (mu U/ml)
            - **BMI**: Body mass index (kg/m²)
            - **DiabetesPedigreeFunction**: Heredity-based diabetes likelihood
            - **Age**: Age in years
            - **Outcome**: Diabetes status (0 = No, 1 = Yes)
            - **measurement_type** (optional): e.g., `fasting`, `post_meal`
            """
        )

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
        pregnancies = c1.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
        glucose = c2.number_input(
            "Plasma Glucose (mg/dL)", 0.0, 300.0, 120.0, help="Measured plasma glucose in mg/dL"
        )
        blood_pressure = c3.number_input(
            "BloodPressure", 0.0, 200.0, 70.0, help="Diastolic blood pressure (mm Hg)"
        )
        skin_thickness = c4.number_input(
            "SkinThickness", 0.0, 100.0, 20.0, help="Triceps skinfold thickness (mm)"
        )

        insulin = c1.number_input("Insulin", 0.0, 900.0, 80.0, help="2-hour serum insulin (mu U/ml)")
        bmi = c2.number_input("BMI", 0.0, 70.0, 28.0, help="Body mass index (kg/m²)")
        dpf = c3.number_input(
            "DiabetesPedigreeFunction", 0.0, 3.0, 0.5, help="Genetic/heredity-based risk factor"
        )
        age = c4.number_input("Age", 10, 100, 33, help="Age in years")

        mtype = c1.selectbox(
            "Measurement Type",
            options=["fasting", "post_meal", "random", "ogtt_2h"],
            index=0,
            help="Context of glucose reading",
        )
        threshold = c2.slider(
            "Decision threshold", 0.0, 1.0, 0.50, 0.01, help="Classify positive if probability ≥ threshold"
        )
        symptomatic = c3.checkbox(
            "Symptoms present (polyuria, polydipsia, weight loss)?",
            value=False,
            help="Used only with random glucose for clinical diagnostic rule",
        )

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
            pred_positive = prob >= float(threshold)

            # Top-line metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability", f"{prob:.2%}")
            m2.metric("Threshold", f"{threshold:.2f}")
            m3.metric("Predicted", "Diabetic" if pred_positive else "Non-diabetic")

            # Gauge visualization
            import plotly.graph_objects as go

            if prob < 0.33:
                gauge_color = "#2ecc71"  # green
            elif prob < 0.66:
                gauge_color = "#f1c40f"  # yellow
            else:
                gauge_color = "#e74c3c"  # red

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100.0,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 33], "color": "#ecf9f1"},
                            {"range": [33, 66], "color": "#fff9e6"},
                            {"range": [66, 100], "color": "#fdecea"},
                        ],
                        "threshold": {
                            "line": {"color": "#34495e", "width": 3},
                            "thickness": 0.75,
                            "value": float(threshold) * 100.0,
                        },
                    },
                    title={"text": "Risk meter"},
                )
            )
            st.plotly_chart(gauge, use_container_width=True)

            # Unified clinical + ML decision
            clinical_inputs = {
                "measurement_type": mtype,
                "glucose": glucose,
                # Optional fields for extension; inputs not collected by default
                "ogtt_2h": None if mtype != "ogtt_2h" else glucose,
                "hba1c": None,
            }
            decision = classify_diabetes(
                inputs=clinical_inputs,
                model_prob=prob,
                threshold=float(threshold),
                symptomatic=bool(symptomatic),
            )

            st.markdown("---")
            st.subheader("Final Decision")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Clinical Category:", f"**{decision['clinical_category']}**")
                st.write("Decision Source:", f"**{decision['decision_source']}**")
            with c2:
                st.write(
                    "Model Risk Probability:", f"**{decision['model_probability']:.1%} ({decision['risk_band']})**"
                )
                st.write("Next Step:", decision["next_step"]) 

            with st.expander("Inputs recap"):
                st.dataframe(df, use_container_width=True)

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


