"""
Healthy Brain Network (HBN) column groupings (demographics, BIA, FitnessGram, PCIAT, …).

pciat_columns() defines leakage features removed before modeling.
"""
from __future__ import annotations

TARGET_COLUMN = "sii"
ID_COLUMNS = ["id"]

# Column lists mirror the Kaggle data dictionary (used in EDA notebooks).
demograficas = ["Basic_Demos-Enroll_Season", "Basic_Demos-Age", "Basic_Demos-Sex"]
clinicas = ["CGAS-Season", "CGAS-CGAS_Score"]

fisicas = [
    "Physical-Season",
    "Physical-BMI",
    "Physical-Height",
    "Physical-Weight",
    "Physical-Waist_Circumference",
    "Physical-Diastolic_BP",
    "Physical-HeartRate",
    "Physical-Systolic_BP",
]

fitness = [
    "Fitness_Endurance-Season",
    "Fitness_Endurance-Max_Stage",
    "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
]

fitnessgram = [
    "FGC-Season",
    "FGC-FGC_CU",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL",
    "FGC-FGC_TL_Zone",
]

bia = [
    "BIA-Season",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_BMC",
    "BIA-BIA_BMI",
    "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "BIA-BIA_ECW",
    "BIA-BIA_FFM",
    "BIA-BIA_FFMI",
    "BIA-BIA_FMI",
    "BIA-BIA_Fat",
    "BIA-BIA_Frame_num",
    "BIA-BIA_ICW",
    "BIA-BIA_LDM",
    "BIA-BIA_LST",
    "BIA-BIA_SMM",
    "BIA-BIA_TBW",
]

actividad = [
    "PAQ_A-Season",
    "PAQ_A-PAQ_A_Total",
    "PAQ_C-Season",
    "PAQ_C-PAQ_C_Total",
]

internet = ["PreInt_EduHx-Season", "PreInt_EduHx-computerinternet_hoursday"]
sueno = ["SDS-Season", "SDS-SDS_Total_Raw", "SDS-SDS_Total_T"]


def pciat_columns() -> list[str]:
    """Internet-use questionnaire — dropped from features (target leakage)."""
    return [
        "PCIAT-Season",
        *[f"PCIAT-PCIAT_{i:02d}" for i in range(1, 21)],
        "PCIAT-PCIAT_Total",
    ]


VARIABLE_CATEGORIES: dict[str, list[str]] = {
    "Identificador (id)": ID_COLUMNS,
    "Target (sii)": [TARGET_COLUMN],
    "Demograficas": demograficas,
    "Clinicas": clinicas,
    "Fisicas": fisicas,
    "Fitness / Resistencia": fitness,
    "FitnessGram": fitnessgram,
    "Biometricas (BIA)": bia,
    "Actividad fisica (PAQ)": actividad,
    "Internet / Habitos": internet,
    "Sueno (SDS)": sueno,
    "PCIAT (NO usar - leakage)": pciat_columns(),
}

KEY_NUMERIC_COLUMNS = [
    "Basic_Demos-Age",
    "Physical-BMI",
    "Physical-Weight",
    "Physical-Height",
    "PreInt_EduHx-computerinternet_hoursday",
    "SDS-SDS_Total_T",
]

ORDINAL_INT_COLUMNS = [
    "Basic_Demos-Sex",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL_Zone",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD_Zone",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_Frame_num",
    "PreInt_EduHx-computerinternet_hoursday",
]
