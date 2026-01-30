import numpy as np
import pandas as pd

# ==========================================
# PROJECT METADATA
# ==========================================
# Project: Loan Data Engineering (NumPy Edition)
# Date: 2025-11-20
# Analyst: Opeyemi Aderibigbe
# Objective: Preprocess loan data using matrix operations for risk modeling.
# ==========================================

def run_numpy_pipeline():
    print("--- Starting Loan Data Engineering Pipeline ---")
    
    # ---------------------------------------------------------
    # 1. LOAD DATA (Exact Notebook Logic)
    # ---------------------------------------------------------
    print("Loading raw data...")
    input_file = "loan-data.csv"
    
    # Load numeric part
    # Note: Using latin1 to match your notebook's implied encoding handling
    raw_data_numeric = np.genfromtxt(input_file, delimiter=';', skip_header=1, autostrip=True, encoding='latin1')
    
    # Load string part (fixed indices from your notebook cells 9 & 10)
    # String Columns Indices: [1, 3, 5, 8, 9, 10, 11, 12]
    columns_strings = np.array([1, 3, 5, 8, 9, 10, 11, 12])
    
    # Numeric Columns Indices: [0, 2, 4, 6, 7, 13]
    columns_numeric = np.array([0, 2, 4, 6, 7, 13])
    
    # Load headers
    header_full = np.genfromtxt(input_file, delimiter=';', skip_footer=raw_data_numeric.shape[0], autostrip=True, encoding='latin1', dtype=str)
    header_strings = header_full[columns_strings]
    header_numeric = header_full[columns_numeric]
    
    # Load string data specifically
    loan_data_strings = np.genfromtxt(input_file, delimiter=';', skip_header=1, autostrip=True, usecols=columns_strings, encoding='latin1', dtype=str)
    
    # Load numeric data specifically
    # Using temporary fill for NaNs just like the notebook
    temporary_fill = np.nanmax(raw_data_numeric) + 1
    loan_data_numeric = np.genfromtxt(input_file, delimiter=';', skip_header=1, autostrip=True, usecols=columns_numeric, filling_values=temporary_fill, encoding='latin1')

    # ---------------------------------------------------------
    # 2. STRING CLEANING (Replicating Cells 28-82)
    # ---------------------------------------------------------
    print("Cleaning string variables...")

    # --- A. Issue Date (Col 0) ---
    # Strip '-15'
    loan_data_strings[:,0] = np.char.strip(loan_data_strings[:,0], "-15")
    # Map months
    months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    for i in range(13):
        loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i], str(i), loan_data_strings[:,0])

    # --- B. Loan Status (Col 1) ---
    status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])
    loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad), '0', '1')

    # --- C. Term (Col 2) ---
    # Strip ' months'
    loan_data_strings[:,2] = np.char.strip(loan_data_strings[:,2], " months")
    # Fill empty with '60'
    loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '', '60', loan_data_strings[:,2])

    # --- D. Grade & Sub-Grade (Cols 3 & 4) ---
    # Fill missing sub_grades using Grade information
    unique_grades = np.unique(loan_data_strings[:,3])
    # The first unique value is likely empty string, so we skip it if needed
    for i in unique_grades:
        if i == '': continue
        loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') & (loan_data_strings[:,3] == i), i + '5', loan_data_strings[:,4])
    
    # Fill remaining empty with 'H1'
    loan_data_strings[:,4] = np.where(loan_data_strings[:,4] == '', 'H1', loan_data_strings[:,4])
    
    # DELETE Grade Column (Index 3) - Crucial Step from Notebook!
    loan_data_strings = np.delete(loan_data_strings, 3, axis=1)
    header_strings = np.delete(header_strings, 3)
    
    # Map Sub-Grade (now at Index 3) to numbers
    keys = list(np.unique(loan_data_strings[:,3]))
    values = list(range(1, len(keys) + 1))
    dict_sub_grade = dict(zip(keys, values))
    
    for i in np.unique(loan_data_strings[:,3]):
        loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i, str(dict_sub_grade[i]), loan_data_strings[:,3])

    # --- E. Verification Status (Col 4) ---
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'), '0', '1')

    # --- F. URL (Col 5) ---
    # DELETE URL Column - Crucial Step from Notebook!
    loan_data_strings = np.delete(loan_data_strings, 5, axis=1)
    header_strings = np.delete(header_strings, 5)

    # --- G. State Address (Col 5 now) ---
    # Region Mapping
    states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
    states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
    states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
    states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])
    
    loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_west), '1', loan_data_strings[:,5])
    loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_south), '2', loan_data_strings[:,5])
    loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_midwest), '3', loan_data_strings[:,5])
    loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_east), '4', loan_data_strings[:,5])
    loan_data_strings[:,5] = np.where(loan_data_strings[:,5] == '', '0', loan_data_strings[:,5])

    # Convert Strings to Int
    loan_data_strings = loan_data_strings.astype(int)

    # ---------------------------------------------------------
    # 3. NUMERIC CLEANING (Replicating Cells 88-101)
    # ---------------------------------------------------------
    print("Imputing numeric variables...")
    
    # Impute Mean
    for i in range(loan_data_numeric.shape[1]):
        column = loan_data_numeric[:, i]
        # Calculate mean excluding the temporary fill value
        valid_vals = column[column != temporary_fill]
        mean_val = np.nanmean(valid_vals) if len(valid_vals) > 0 else 0
        
        loan_data_numeric[:,i] = np.where(column == temporary_fill, mean_val, column)

    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    print("Exporting to Excel...")
    
    # Stack Numeric and Cleaned Strings
    loan_data = np.hstack((loan_data_numeric, loan_data_strings))
    
    # Sort by ID (Column 0)
    loan_data = loan_data[np.argsort(loan_data[:,0])]
    
    # Combine Headers
    full_header = np.concatenate((header_numeric, header_strings))
    
    # Save as Excel (Requirements: "Document for Excel User")
    output_file = 'Clean_Loan_Data_Risk_Ready.xlsx'
    df_final = pd.DataFrame(loan_data, columns=full_header)
    df_final.to_excel(output_file, index=False)
    
    print(f"Success! Created {output_file}")

if __name__ == "__main__":
    run_numpy_pipeline()