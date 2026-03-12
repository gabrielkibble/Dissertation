import pandas as pd
import os

# --- CONFIGURATION ---
input_files = ['Data_exfiltration.csv', 'Keylogging.csv']
output_filename = 'Theft_Full.csv'

def clean_and_merge():
    combined_data = []

    for file in input_files:
        if not os.path.exists(file):
            print(f"Skipping {file}: File not found.")
            continue
            
        print(f"Processing {file}...")
        
        # 1. READ FILE (Try ';' first, then ',')
        try:
            df = pd.read_csv(file, sep=';', low_memory=False)
            if df.shape[1] < 2: # If it failed to separate, try comma
                df = pd.read_csv(file, sep=',', low_memory=False)
        except:
            df = pd.read_csv(file, sep=',', low_memory=False)

        # 2. REMOVE THE EXTRA 'record' COLUMN
        if 'record' in df.columns:
            print(f"  - Dropping 'record' column...")
            df = df.drop(columns=['record'])

        combined_data.append(df)

    # 4. COMBINE AND SAVE
    if combined_data:
        df_final = pd.concat(combined_data, ignore_index=True)
        
        # Save as standard CSV (comma separated)
        df_final.to_csv(output_filename, index=False)
        
        print(f"\nSuccess! Created '{output_filename}' with {len(df_final)} rows.")
        print(f"Columns: {list(df_final.columns)}")
    else:
        print("\nError: No data loaded.")

if __name__ == "__main__":
    clean_and_merge()