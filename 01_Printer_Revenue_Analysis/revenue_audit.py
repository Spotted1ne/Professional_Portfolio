import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PROJECT METADATA
# ==========================================
# Project: Global Revenue Risk Audit
# Date: 2025-12-10
# Analyst: Opeyemi Aderibigbe
# Objective: Reconcile regional sales data to identify concentration risks.
# ==========================================

def run_audit_pipeline():
    # ---------------------------------------------------------
    # 1. DATA INGESTION
    # ---------------------------------------------------------
    input_file = 'Global_sales_ledger.csv'
    
    try:
        # LOGIC LOG: Loading raw sales ledger. 
        # Source file is standard CSV.
        df = pd.read_csv(input_file)
        print(f"[Success] Loaded ledger: {input_file}")
    except FileNotFoundError:
        print(f"[Error] File '{input_file}' not found. Please ensure it is in the folder.")
        return

    # ---------------------------------------------------------
    # 2. DATA CLEANING & RECONCILIATION
    # ---------------------------------------------------------
    
    # LOGIC LOG: The export tool generated empty 'Unnamed' columns.
    # ACTION: Dropping them to ensure clean schema for analysis.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # LOGIC LOG: Data is in 'Wide' format (Regions as columns).
    # ACTION: Pivoting to 'Long' format using melt(). This is required to 
    # normalize the regions into a single dimension for the revenue model.
    df_long = df.melt(
        id_vars=['Date'], 
        value_vars=['International', 'Domestic', 'Services'],
        var_name='Region', 
        value_name='Revenue'
    )

    # ---------------------------------------------------------
    # 3. RISK ANALYSIS
    # ---------------------------------------------------------
    
    # LOGIC LOG: Aggregating total revenue to calculate market share.
    revenue_summary = df_long.groupby('Region')['Revenue'].sum().reset_index()

    # Calculation: Deriving percentage share per region.
    total_revenue = revenue_summary['Revenue'].sum()
    revenue_summary['Share_Pct'] = (revenue_summary['Revenue'] / total_revenue) * 100
    
    # Sorting to identify the largest dependency.
    revenue_summary = revenue_summary.sort_values('Share_Pct', ascending=False)

    print("\n--- Revenue Concentration Report (Dec 2025) ---")
    print(revenue_summary.round(2))

    # ---------------------------------------------------------
    # 4. VISUALIZATION
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # LOGIC LOG: Visualizing the revenue distribution to highlight disparities.
    sns.barplot(data=revenue_summary, x='Region', y='Share_Pct', palette='viridis')

    # LOGIC LOG: Plotting the 60% threshold line.
    # Rationale: The risk assessment identifies >60% dependency on a single market
    # as a critical vulnerability.
    plt.axhline(60, color='red', linestyle='--', linewidth=2, label='Risk Threshold (60%)')

    plt.title('Revenue Dependency Risk Analysis', fontsize=14, fontweight='bold')
    plt.ylabel('Share of Total Revenue (%)', fontsize=12)
    plt.xlabel('Market Region', fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Archiving the visual evidence
    output_chart = 'revenue_risk_chart.png'
    plt.savefig(output_chart)
    print(f"\n[Success] Chart archived as '{output_chart}'")

if __name__ == "__main__":
    run_audit_pipeline()