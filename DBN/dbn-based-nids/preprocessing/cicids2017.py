import pandas as pd
import numpy as np
import glob
import os
import sys

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Use a more robust way to find the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Replace your old DATA_DIR line with these:
SCRIPT_PATH = os.path.abspath(__file__)
PREPROCESSING_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.dirname(PREPROCESSING_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

print(f"DEBUG: Data Directory is set to: {DATA_DIR}")

class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.labels = None # Changed from label to labels to match usage later

    def read_data(self):
        search_path = os.path.join(self.data_path, 'raw', '*.csv')
        filenames = glob.glob(search_path)
        print(f"[1/8] Found {len(filenames)} files in {search_path}")
        
        if not filenames:
            print("!!! ERROR: No CSV files found. Check your data/raw folder.")
            sys.exit(1)

        datasets = []
        for filename in filenames:
            print(f"      Reading {os.path.basename(filename)}...")
            df = pd.read_csv(filename, encoding='cp1252')
            # Clean column names immediately
            df.columns = [self._clean_column_name(col) for col in df.columns]
            datasets.append(df)

        print("      Concatenating datasets...")
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        
        # Safe drop for the duplicate header column
        if 'fwd_header_length.1' in self.data.columns:
            self.data.drop(labels=['fwd_header_length.1'], axis=1, inplace=True)

    def _clean_column_name(self, column):
        return column.strip().replace('/', '_').replace(' ', '_').lower()

    def remove_duplicate_values(self):
        print("[2/8] Removing duplicates...")
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        print("[3/8] Removing missing values...")
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        print("[4/8] Removing infinite values...")
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)
        self.data.dropna(axis=0, how='any', inplace=True)

    def remove_constant_features(self, threshold=0.01):
        print("[5/8] Removing constant features...")
        data_std = self.data.std(numeric_only=True)
        # FIX: iterateitems() -> items() for Pandas 3.0
        constant_features = [column for column, std in data_std.items() if std < threshold]
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):
        print("[6/8] Removing highly correlated features (this may take a minute)...")
        numeric_data = self.data.select_dtypes(include=[np.number])
        data_corr = numeric_data.corr()
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        print("[7/8] Grouping attack labels...")
        attack_group = {
            'BENIGN': 'Benign', 'PortScan': 'PortScan', 'DDoS': 'DoS/DDoS',
            'DoS Hulk': 'DoS/DDoS', 'DoS GoldenEye': 'DoS/DDoS',
            'DoS slowloris': 'DoS/DDoS', 'DoS Slowhttptest': 'DoS/DDoS',
            'Heartbleed': 'DoS/DDoS', 'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force', 'Bot': 'Botnet ARES',
            'Web Attack  Brute Force': 'Web Attack',
            'Web Attack  Sql Injection': 'Web Attack',
            'Web Attack  XSS': 'Web Attack',
            'Infiltration': 'Infiltration'
        }
        # Use .get() to avoid KeyError if labels have trailing spaces
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group.get(x.strip(), 'Other'))
        
    def train_valid_test_split(self):
        print("[8/8] Splitting data into Train/Val/Test...")
        self.labels = self.data['label_category']
        
        # 1. DROP IDENTITY COLUMNS (Prevents the 5TB Memory Error)
        cols_to_drop = ['label', 'label_category', 'flow_id', 'source_ip', 'destination_ip', 'timestamp', 'external_ip']
        existing_drops = [c for c in cols_to_drop if c in self.data.columns]
        self.features = self.data.drop(labels=existing_drops, axis=1)

        # 2. PERFORM SPLIT
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=42, stratify=self.labels
        )
        
        val_ratio = self.validation_size / (self.validation_size + self.testing_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
        )
    
        # 3. THE CRITICAL RETURN (This fixes the TypeError)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    
    def scale(self, training_set, validation_set, testing_set):
        print("      Scaling and Encoding features...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set
        
        # Identify columns
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(include=["number"]).columns

        # Setup Transformer
        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
        ])

        # Transform features
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)
        
        # Get feature names safely
        try:
            new_cols = preprocessor.get_feature_names_out()
        except:
            new_cols = [f"f_{i}" for i in range(X_train_proc.shape[1])]

        # Reconstruct DataFrames
        X_train_final = pd.DataFrame(X_train_proc, columns=new_cols)
        X_val_final = pd.DataFrame(X_val_proc, columns=new_cols)
        X_test_final = pd.DataFrame(X_test_proc, columns=new_cols)

        # Preprocess the labels
        le = LabelEncoder()
        y_train_final = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        y_val_final = pd.DataFrame(le.transform(y_val), columns=["label"])
        y_test_final = pd.DataFrame(le.transform(y_test), columns=["label"])

        # THE CRITICAL RETURN (Names must match exactly what was defined above)
        return (X_train_final, y_train_final), (X_val_final, y_val_final), (X_test_final, y_test_final)
# REMOVE the if __name__ == "__main__": line entirely
# Just start the code at the edge of the screen (no indentation)

print("--- PIPELINE STARTING (FORCED) ---")
proc = CICIDS2017Preprocessor(DATA_DIR, 0.6, 0.2, 0.2)
proc.read_data()
proc.remove_duplicate_values()
proc.remove_missing_values()
proc.remove_infinite_values()
proc.remove_constant_features()
proc.remove_correlated_features()
proc.group_labels()
proc.data = proc.data[proc.data['label_category'] != 'Infiltration']
sets = proc.train_valid_test_split()
(XT, YT), (XV, YV), (XTe, YTe) = proc.scale(*sets)

print("Saving...")
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATA_DIR, 'processed', folder), exist_ok=True)

XT.to_pickle(os.path.join(DATA_DIR, 'processed/train/train_features.pkl'))
XV.to_pickle(os.path.join(DATA_DIR, 'processed/val/val_features.pkl'))
XTe.to_pickle(os.path.join(DATA_DIR, 'processed/test/test_features.pkl'))
YT.to_pickle(os.path.join(DATA_DIR, 'processed/train/train_labels.pkl'))
YV.to_pickle(os.path.join(DATA_DIR, 'processed/val/val_labels.pkl'))
YTe.to_pickle(os.path.join(DATA_DIR, 'processed/test/test_labels.pkl'))
print("--- FINISHED ---")