# Oncology Claims Data — AI Insight Engine

Automated clinical insight pipeline for lung cancer patient analysis 
using Real World Evidence (RWE) claims data.

## What this does
- Identifies patient cohorts from claims data (ICD-10, NDC, CPT, LOINC)
- Runs SQL queries to find high-risk patients
- Generates plain-English clinical summaries using Google Gemini AI

## Key Findings from Analysis
- 65% patients diagnosed at Stage III or Stage IV
- Identified 53 Stage IV patients on Immunotherapy (Pembrolizumab, Atezolizumab)
- Flagged 37 Medicare patients with high CEA levels (avg 35.12 ng/mL)
- Recommended immediate multidisciplinary tumor board review for high-risk cohort

## Tech Stack
- Python
- DuckDB (SQL on DataFrames)
- Pandas
- Google Gemini AI API

## Data
Synthetic claims data built on real-world schema:
- ICD-10 diagnosis codes (C34.x lung cancer)
- NDC drug codes (Carboplatin, Pembrolizumab, Osimertinib)
- CPT procedure codes
- LOINC lab codes (PD-L1, EGFR, CEA)
- Medicare and Commercial payer mix

## How to Run
1. Clone the repo
2. Install dependencies: pip install duckdb pandas google-genai numpy
3. Set your API key: $env:GEMINI_API_KEY="your-key-here"
4. Run: python lung_cancer.py

## Author
Healthcare Data Analyst | RWE | Clinical Data Automation
