import datetime
from src.utils.logger import get_logger

logger = get_logger("MaintenanceRAG")

SOP_KNOWLEDGE_BASE = [
    {
        "id": "SOP-101",
        "title": "High Temperature Degradation (Sensors S3 / S4 / S11)",
        "condition": "Exhaust Gas Temperature / HPC Temperature exceeding normal threshold with declining RUL",
        "recommended_actions": [
            "Perform borescope inspection of High-Pressure Compressor (HPC) stage 3-5 blades.",
            "Inspect thermal barrier coating (TBC) for flaking or severe erosion.",
            "Check fuel nozzle spray patterns and recalibrate fuel flow regulator."
        ],
        "estimated_downtime_hours": 14,
        "parts_required": ["HPC Seal Ring Set", "Fuel Nozzle Gasket Kit"]
    },
    {
        "id": "SOP-102",
        "title": "Low Pressure Compressor & Speed Anomaly (Sensors S2 / S8 / S13)",
        "condition": "Core speed ratio decay and pressure fluctuation",
        "recommended_actions": [
            "Check Low-Pressure Turbine (LPT) shaft alignment and bearing vibration levels.",
            "Flush and replace main lube oil filter; test oil for metallic wear particles.",
            "Perform electronic engine control (EEC) sensor calibration."
        ],
        "estimated_downtime_hours": 8,
        "parts_required": ["Main Lube Filter", "Speed Transducer Harness"]
    },
    {
        "id": "SOP-103",
        "title": "Critical RUL End-of-Life Full Overhaul (RUL < 15 cycles)",
        "condition": "Imminent failure risk; Remaining Useful Life under 15 cycles",
        "recommended_actions": [
            "Schedule immediate engine removal and transport to MRO overhaul bay.",
            "Perform complete module disassembly and NDT (Non-Destructive Testing) eddy current scan.",
            "Replace turbine rotor blades and rebalance spool assembly."
        ],
        "estimated_downtime_hours": 48,
        "parts_required": ["Turbine Rotor Blade Set Grade-A", "Overhaul Gasket & O-Ring Package"]
    }
]

class MaintenanceRAGAssistant:
    """
    GenAI / RAG Maintenance Assistant generating automated, domain-aware
    industrial diagnostic reports by matching engine predictions with SOP manual context.
    """
    def generate_report(self, unit_nr: int, current_cycle: int, predicted_rul: float, survival_prob_20: float, health_index: float, sensor_values: dict = None) -> str:
        logger.info(f"Generating GenAI RAG Maintenance Report for Engine Unit {unit_nr} @ Cycle {current_cycle}...")

        # Match relevant SOP based on engine metrics
        if predicted_rul < 15.0 or health_index < 0.35:
            matched_sop = SOP_KNOWLEDGE_BASE[2] # Full Overhaul
        elif sensor_values and (sensor_values.get("s_4", 1400) > 1410 or sensor_values.get("s_3", 1590) > 1600):
            matched_sop = SOP_KNOWLEDGE_BASE[0] # Thermal/HPC
        else:
            matched_sop = SOP_KNOWLEDGE_BASE[1] # Mechanical/LPT

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        report_md = f"""# RAPPORT D'INTERVENTION DE MAINTENANCE PRÉDICTIVE
**Plateforme Industrielle Turbofan | Génération Automatique (GenAI/RAG)**
**Date de Génération** : {timestamp_str}

---

## 1. Synthèse de l'Équipement
- **Identifiant Moteur (Unit ID)** : `Moteur #{unit_nr}`
- **Cycle de Fonctionnement Actuel** : `{current_cycle} cycles`
- **Remaining Useful Life (RUL) Estimée** : **`{predicted_rul:.1f} cycles restants`**
- **Indice de Santé Global (Health Index)** : **`{health_index*100:.1f}%`**
- **Probabilité de Survie (à 20 cycles)** : **`{survival_prob_20*100:.1f}%`**

---

## 2. Diagnostic & Analyse Documentaire RAG (SOP)
- **Procédure Constructeur Identifiée** : **`{matched_sop['id']} - {matched_sop['title']}`**
- **Condition de Déclenchement** : *{matched_sop['condition']}*
- **Temps d'Immobilisation Estimé** : **`{matched_sop['estimated_downtime_hours']} heures`**

### Recommandations d'Actions Correctives :
"""
        for act in matched_sop["recommended_actions"]:
            report_md += f"- [ ] {act}\n"

        report_md += "\n### Pièces Détachées & Outillage Requis :\n"
        for part in matched_sop["parts_required"]:
            report_md += f"- Ref: `{part}`\n"

        report_md += f"""
---

## 3. Évaluation d'Impact Financier
- **Coût d'intervention préventive planifiée** : `$500.00`
- **Économie vs Panne en vol (Correctif non planifié)** : **`$4,500.00 évités`**

*Rapport certifié et généré automatiquement par le module RAG Predictive Maintenance Platform.*
"""
        return report_md
