import duckdb
import pandas as pd
import numpy as np
from google import genai

# 1. GEMINI SETUP
import os
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. DATA
np.random.seed(42)

ICD10_CODES = {
    "C34.10": "Lung cancer, upper lobe",
    "C34.11": "Lung cancer, upper lobe right",
    "C34.12": "Lung cancer, upper lobe left",
    "C34.30": "Lung cancer, lower lobe",
    "C34.90": "Lung cancer, unspecified",
    "Z87.891": "Personal history of lung cancer",
}

NDC_DRUGS = {
    "00015054841": "Carboplatin",
    "00015054836": "Cisplatin",
    "00004014750": "Pembrolizumab",
    "50242006401": "Atezolizumab",
    "00310055060": "Osimertinib",
    "00378935093": "Crizotinib",
    "00002751559": "Pemetrexed",
}

CPT_CODES = {
    "99213": "Office visit - established",
    "99214": "Office visit - complex",
    "71046": "Chest X-ray",
    "71250": "CT Chest",
    "96413": "Chemo infusion - first hour",
    "96415": "Chemo infusion - additional hour",
}

LOINC_CODES = {
    "85319-2": "PD-L1 expression",
    "55233-1": "EGFR mutation",
    "72518-4": "ALK gene rearrangement",
    "14804-9": "LDH",
    "2532-0":  "CEA",
}

rows = []
for i in range(1, 101):
    member = f"M{str(i).zfill(4)}"
    age    = np.random.randint(45, 85)
    gender = np.random.choice(["M", "F"], p=[0.6, 0.4])
    biz    = np.random.choice(["Medicare", "Commercial"], p=[0.65, 0.35])
    stage  = np.random.choice(["Stage I","Stage II","Stage III","Stage IV"], p=[0.15,0.20,0.25,0.40])
    icd    = np.random.choice(list(ICD10_CODES.keys()))

    for _ in range(np.random.randint(3, 12)):
        svc_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(np.random.randint(0, 365)))
        ndc      = np.random.choice(list(NDC_DRUGS.keys())) if stage in ["Stage III","Stage IV"] else None
        cpt      = np.random.choice(list(CPT_CODES.keys()))
        loinc    = np.random.choice(list(LOINC_CODES.keys()))
        loinc_desc = LOINC_CODES[loinc]

        if loinc_desc == "PD-L1 expression":
            lab_val = round(np.random.uniform(0, 100), 1)
        elif loinc_desc == "LDH":
            lab_val = round(np.random.uniform(100, 500), 1)
        elif loinc_desc == "CEA":
            lab_val = round(np.random.uniform(0.5, 50), 2)
        else:
            lab_val = None

        rows.append({
            "member_id":      member,
            "age":            age,
            "gender":         gender,
            "business_ln_cd": biz,
            "stage":          stage,
            "icd10":          icd,
            "icd10_desc":     ICD10_CODES[icd],
            "ndc":            ndc,
            "drug_name":      NDC_DRUGS.get(ndc),
            "cpt":            cpt,
            "cpt_desc":       CPT_CODES[cpt],
            "loinc":          loinc,
            "loinc_desc":     loinc_desc,
            "lab_value":      lab_val,
            "service_date":   svc_date,
        })

df = pd.DataFrame(rows).sort_values(["member_id","service_date"]).reset_index(drop=True)
print(f"[OK] Data ready -- {len(df)} records, {df['member_id'].nunique()} patients")

# 3. SQL COHORT QUERIES
cohort_1 = duckdb.sql("""
    SELECT 
        member_id, age, gender, business_ln_cd, drug_name,
        COUNT(*) as total_visits
    FROM df
    WHERE stage = 'Stage IV'
      AND drug_name IN ('Pembrolizumab', 'Atezolizumab')
    GROUP BY member_id, age, gender, business_ln_cd, drug_name
    ORDER BY total_visits DESC
""").df()

print(f"\n[DATA] Stage IV Immunotherapy Cohort: {len(cohort_1)} patients")
print(cohort_1.head())

cohort_2 = duckdb.sql("""
    SELECT member_id, age, stage,
           ROUND(AVG(lab_value), 2) as avg_cea
    FROM df
    WHERE business_ln_cd = 'Medicare'
      AND loinc_desc = 'CEA'
      AND lab_value > 20
    GROUP BY member_id, age, stage
    ORDER BY avg_cea DESC
""").df()

print(f"\n[DATA] Medicare High CEA Patients: {len(cohort_2)} patients")
print(cohort_2.head())

drug_by_stage = duckdb.sql("""
    SELECT stage, drug_name,
           COUNT(DISTINCT member_id) as patients
    FROM df
    WHERE drug_name IS NOT NULL
    GROUP BY stage, drug_name
    ORDER BY stage, patients DESC
""").df()

print(f"\n[DATA] Drug Distribution by Stage:")
print(drug_by_stage)

# 4. AI ANALYSIS
print("\n[AI] Sending to Gemini...")

summary = f"""
Lung Cancer Claims Data Analysis:

Total patients: {df['member_id'].nunique()}
Stage distribution: {df.drop_duplicates('member_id')['stage'].value_counts().to_dict()}
Business line: {df.drop_duplicates('member_id')['business_ln_cd'].value_counts().to_dict()}

Stage IV Immunotherapy cohort: {len(cohort_1)} patients
- Drugs used: {cohort_1['drug_name'].value_counts().to_dict()}
- Avg age: {round(cohort_1['age'].mean(), 1)}

Medicare High CEA patients: {len(cohort_2)} patients
- Avg CEA level: {round(cohort_2['avg_cea'].mean(), 2)}
- Stage breakdown: {cohort_2['stage'].value_counts().to_dict()}
"""

prompt = f"""
You are a clinical data analyst specializing in oncology Real World Evidence.

Analyze this lung cancer claims data and provide:
1. Key patient population insights
2. Treatment pattern observations
3. Which cohort needs immediate clinical attention and why
4. One specific recommendation for the medical team

Data:
{summary}

Be specific. Use numbers. Write in plain English that a doctor can understand.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\n" + "="*50)
print("CLINICAL INSIGHT REPORT")
print("="*50)
print(response.text)