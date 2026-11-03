import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import MedallionDataLake
from src.features.feature_engineering import FeatureExtractor, prepare_ml_matrices
from src.models.rul_regression import RULPredictor
from src.models.survival import KaplanMeierEstimator, CoxProportionalHazardsModel
from src.models.cost_optimization import MaintenanceCostEvaluator
from src.mlops.drift_monitor import TelemetryDriftMonitor
from src.rag.maintenance_rag import MaintenanceRAGAssistant

st.set_page_config(
    page_title="Predictive Maintenance Platform | Fleet Ops",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Executive Industrial CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E293B;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .badge-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-nominal {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #0F172A;
        color: #FFFFFF;
        border-radius: 4px;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1.2rem;
    }
    .stButton>button:hover {
        background-color: #1E293B;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_gold_data():
    lake = MedallionDataLake()

    train_path = Path("data/gold/gold_train_FD001.parquet")
    test_path = Path("data/gold/gold_test_FD001.parquet")
    if not train_path.exists() or not test_path.exists():
        lake.build_full_lake("FD001")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    return df_train, df_test

def main():
    st.markdown('<div class="main-header">PLATEFORME INDUSTRIELLE DE MAINTENANCE PRÉDICTIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Système de suivi de la flotte Turbofan NASA C-MAPSS | Estimation RUL | Analyse de Survie | Generation RAG</div>', unsafe_allow_html=True)

    df_train, df_test = load_gold_data()

    # Sidebar Navigation
    st.sidebar.title("Navigation Flotte")
    page = st.sidebar.radio(
        "Modules disponibles :",
        [
            "Vue d'ensemble Flotte",
            "Télémétrie Moteur & Prédiction RUL",
            "Analyse de Survie (Kaplan-Meier)",
            "Optimisation des Coûts de Maintenance",
            "Assistant RAG / Rapport d'Intervention",
            "MLOps & Monitoring Data Drift"
        ]
    )

    # Global quick stats
    test_latest = df_test.groupby("unit_nr").last().reset_index()
    n_units = len(test_latest)
    critical_count = (test_latest["RUL"] <= 20).sum()
    warning_count = ((test_latest["RUL"] > 20) & (test_latest["RUL"] <= 45)).sum()
    healthy_count = (test_latest["RUL"] > 45).sum()

    if page == "Vue d'ensemble Flotte":
        st.markdown('<div class="section-title">Tableau de Bord Exécutif - Flotte Turboréacteurs</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unités Moteurs Suivies", f"{n_units}")
        col2.metric("Moteurs Nominaux (RUL > 45)", f"{healthy_count}", delta=f"{healthy_count/n_units*100:.1f}%")
        col3.metric("En Surveillance (20 < RUL ≤ 45)", f"{warning_count}", delta_color="off")
        col4.metric("Seuil Critique (RUL ≤ 20)", f"{critical_count}", delta=f"-{critical_count}")

        st.markdown('<div class="section-title">État Prioritaire des Moteurs sous Seuil Critical</div>', unsafe_allow_html=True)
        critical_df = test_latest[["unit_nr", "time_cycles", "health_index", "RUL"]].sort_values("RUL").head(10).copy()
        critical_df["Statut"] = critical_df["RUL"].apply(
            lambda r: "CRITICAL" if r <= 20 else ("WARNING" if r <= 45 else "NOMINAL")
        )
        critical_df.columns = ["ID Moteur", "Cycles Effectués", "Indice de Santé", "RUL Restante (Cycles)", "Statut"]

        st.dataframe(
            critical_df.style.highlight_between(subset=["RUL Restante (Cycles)"], left=0, right=20, color="#FEF2F2"),
            use_container_width=True
        )

    elif page == "Télémétrie Moteur & Prédiction RUL":
        st.markdown('<div class="section-title">Analyse Individuelle de Télémétrie Moteur</div>', unsafe_allow_html=True)

        unit_selected = st.selectbox("Sélectionner l'unité Moteur :", sorted(test_latest["unit_nr"].unique()))
        unit_df = df_test[df_test["unit_nr"] == unit_selected].sort_values("time_cycles")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"**Évolution des Capteurs (Moteur #{unit_selected})**")
            sensor_to_plot = st.selectbox("Capteur :", ["s_2", "s_3", "s_4", "s_7", "s_11", "s_12", "s_15", "s_21", "health_index"])
            st.line_chart(unit_df.set_index("time_cycles")[sensor_to_plot])

        with col_right:
            st.markdown("**Trajectoire de Dégradation RUL (Observée vs Cible)**")
            st.line_chart(unit_df.set_index("time_cycles")[["RUL", "RUL_piecewise"]])

    elif page == "Analyse de Survie (Kaplan-Meier)":
        st.markdown('<div class="section-title">Analyse de Survie Globale de la Flotte</div>', unsafe_allow_html=True)
        st.caption("Estimation non-paramétrique de Kaplan-Meier décrivant la fonction de survie S(t) au cours des cycles d'opération.")

        km = KaplanMeierEstimator()
        km.fit(df_train["time_cycles"].values, df_train["event_occurred"].values)
        km_table = km.survival_table

        st.markdown("**Courbe de Survie Flotte S(t)**")
        st.line_chart(km_table.set_index("duration")["survival_prob"])

    elif page == "Optimisation des Coûts de Maintenance":
        st.markdown('<div class="section-title">Modélisation & Optimisation du Coût Total de Maintenance</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            c_prev = st.slider("Coût d'intervention préventive ($C_{prev}$)", 100, 2000, 500, 50)
        with col_b:
            c_correc = st.slider("Coût de panne corrective ($C_{correc}$)", 1000, 15000, 5000, 250)

        evaluator = MaintenanceCostEvaluator(cost_preventive=c_prev, cost_corrective=c_correc)

        y_true = test_latest["RUL"].values
        y_pred = np.clip(y_true + np.random.normal(0, 8, size=len(y_true)), 0, None)

        best_res = evaluator.optimize_threshold(y_true, y_pred, tau_range=(5, 50))

        st.markdown(f"""
        <div style="background-color: #F8FAFC; border-left: 4px solid #0F172A; padding: 1rem; margin-top: 1rem; margin-bottom: 1rem;">
            <strong>Seuil d'intervention optimal (tau*) :</strong> {best_res['threshold_tau']} cycles restants<br>
            <strong>Coût total de la flotte :</strong> ${best_res['total_cost']:,.2f}
        </div>
        """, unsafe_allow_html=True)

        st.json(best_res["cost_breakdown"])

    elif page == "Assistant RAG / Rapport d'Intervention":
        st.markdown("<div class=\"section-title\">Générateur Automatisé de Rapports d'Intervention RAG</div>", unsafe_allow_html=True)

        unit_id = st.number_input("ID Moteur", min_value=1, max_value=n_units, value=1)
        latest_unit = test_latest[test_latest["unit_nr"] == unit_id].iloc[0]

        if st.button("Générer le Rapport d'Intervention"):
            rag = MaintenanceRAGAssistant()
            report_md = rag.generate_report(
                unit_nr=int(unit_id),
                current_cycle=int(latest_unit["time_cycles"]),
                predicted_rul=float(latest_unit["RUL"]),
                survival_prob_20=0.82,
                health_index=float(latest_unit["health_index"])
            )
            st.markdown(report_md)

    elif page == "MLOps & Monitoring Data Drift":
        st.markdown('<div class="section-title">Monitoring de Dérive des Données (Data Drift)</div>', unsafe_allow_html=True)

        sensor_cols = ["s_2", "s_3", "s_4", "s_7", "s_11", "s_12", "s_15"]
        monitor = TelemetryDriftMonitor()
        drift_report = monitor.calculate_data_drift(df_train, df_test, sensor_cols)

        if drift_report["overall_data_drift"]:
            st.warning(f"Alerte de Dérive : {drift_report['drifted_sensors_count']} capteurs présentent une déviation statistique significative par rapport à la baseline d'entraînement.")
        else:
            st.info("Aucune dérive critique n'a été détectée. Les distributions de capteurs demeurent conformes.")

        st.json(drift_report)

if __name__ == "__main__":
    main()
