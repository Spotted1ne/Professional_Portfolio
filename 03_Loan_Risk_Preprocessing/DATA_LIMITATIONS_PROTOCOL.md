# Data Limitations & Assumptions Protocol
**Project:** Loan Risk Preprocessing (NumPy Edition)
**Date:** November 2025
**Author:** Opeyemi Aderibigbe

## 1. Overview
This protocol documents the cleaning logic and assumptions applied to the Lending Club dataset (`loan-data.csv`). It serves as a transparent record for stakeholders to understand how data quality gaps were handled prior to risk modeling.

## 2. Key Assumptions & Imputations
* **Missing Term Lengths:** Any loan with a missing term was imputed as **60 months** (worst-case scenario) to avoid underestimating risk exposure.
* **Financial Imputation:** Missing numeric fields (e.g., Funded Amount, Payment History) were imputed using the **column mean**. This ensures dataset completeness without skewing the statistical distribution.
* **Region Mapping:** State abbreviations were mapped to four census regions (West, South, Midwest, East) to allow for broader geographic risk analysis. States with missing values were assigned a placeholder code ('0').
* **Risk Classification:** Loan Status was binary encoded:
    * **1 (Good):** "Current", "Fully Paid", "In Grace Period"
    * **0 (Bad):** "Charged Off", "Default", "Late"

## 3. Data Exclusions (Feature Selection)
To improve processing efficiency and model relevance, the following features were removed:
* **URL:** Removed as it contains no predictive value for credit risk.
* **Grade:** Removed because it is redundant; `sub_grade` provides more granular risk information and was retained.

## 4. Execution
These protocols are implemented programmatically in `loan_data_engineering.py` using NumPy for vectorized processing.