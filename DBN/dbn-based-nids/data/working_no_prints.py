# ==========
# Imports
# ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from hmmlearn.hmm import CategoricalHMM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time



# ============
# File Paths
# ============

# have cherry picked these as they contain some of all, still maybe look for another file with theft in later
FULL_SOURCE_FILES = ["UNSW-NB15_1.csv",
                     "UNSW-NB15_2.csv",
                     "UNSW-NB15_3.csv",
                     "UNSW-NB15_4.csv"
                     ]

BALANCED_FILE = "balanced_NB15_iot.csv"                  # The output file we create/use
OUTPUT_SEQ_FILE = "novel_attack_sequences2.npy"

# Need all of it
IMBALANCED_FILES = FULL_SOURCE_FILES

# The raw UNSW-NB15 files do not have headers. We must define them explicitly.
RAW_COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", 
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", 
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean", 
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sinpkt", 
    "dinpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", 
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", 
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", 
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
]



# ========================
# Toggles and Parameters
# ========================

# Toggles for what code is run
RECREATE_BALANCED_DATA = False  # Set to False if "balanced_bot_iot.csv" already exists
RECREATE_SEQUENCES = True      # Set to False if "novel_attack_sequences.npy" already exists

# Parameters
TARGET_SAMPLES_PER_CLASS = 300
SEQ_LENGTH = 20 # have a graph printed now to justify this, maybe change to a little higher
STRIDE = 10 # should be half of seq_length for overlap, if above changes, change this
TEMPERATURE = 1.2 # see google doc for justification for why its not that large


FEATURES = [
    "dur",                      # Duration
    "spkts", "dpkts",           # Packet counts
    "sbytes", "dbytes",         # Volume
    "sload", "dload",           # Speed (Bits per second)
    "smean", "dmean",           # Average packet sizes (Note: 'smean', not 'smeansz')
    "sjit", "djit",             # Jitter (mSec)
    "sinpkt", "dinpkt"          # Interpacket arrival (Note: 'sinpkt', not 'Sintpkt')
]



# ============================
# Building Balanced Dataset
# ============================

# Only end up with: {'DDoS': 3000, 'DoS': 3000, 'Reconnaissance': 3000, 'Theft': 79, 'Normal': 477}
# but this is fine, the fact that 'theft' is rare justifies using gen AI and not just a classifer
# normal traffic number is fine as its enough for context and not critical
# will however have to synthetically oversample theft to trick my CVAE into not treating it as noise
def create_balanced_dataset(input_path_list, output_path, target_samples):
    """
    Scans a list of large datasets and extracts a stratified sample of each category.
    """
    print("\n--- Creating Balanced Dataset from Multiple Files ---")
    
    # Buckets persist across all files
    # Matches the categories defined in the BoT-IoT dataset
    data_buckets = {
        "Normal": [], "Generic": [], "Exploits": [], "Fuzzers": [], 
        "DoS": [], "Reconnaissance": [], "Analysis": [], 
        "Backdoors": [], "Shellcode": [], "Worms": []
    }
    
    chunk_size = 100000
    all_done = False # flag to break the outer loop if we finish early

    # Iterate through every file in the list
    for file_path in input_path_list:
        if all_done:
            break
            
        print(f"Scanning file: {file_path}...")

        try:
            # Process chunks within the current file
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, names=RAW_COLUMN_NAMES, header=None):
                
                # 1. Standardise the column name to 'category'
                if 'attack_cat' in chunk.columns:
                    chunk.rename(columns={'attack_cat': 'category'}, inplace=True)
                
                if 'category' not in chunk.columns:
                    print(f"Warning: Column 'category' not found in {file_path}. Skipping chunk.")
                    continue

                # 2. Fix the UNSW-NB15 specific blank cells and whitespaces
                chunk['category'] = chunk.apply(
                    lambda row: 'Normal' if pd.isna(row['category']) and row['Label'] == 0 else str(row['category']).strip(), 
                    axis=1
                )

                # Fill buckets
                for cat in data_buckets.keys():
                    current_count = sum([len(df) for df in data_buckets[cat]])
                    
                    if current_count < target_samples:
                        subset = chunk[chunk["category"] == cat]
                        if len(subset) > 0:
                            needed = target_samples - current_count
                            data_buckets[cat].append(subset.head(needed))

                # Check if ALL buckets are full
                total_counts = {k: sum([len(df) for df in v]) for k, v in data_buckets.items()}
                
                # Progress update
                print(f"Status: {total_counts}")

                if all(c >= target_samples for c in total_counts.values()):
                    print("All buckets full! Stopping scan.")
                    all_done = True
                    break
        
        except Exception as e:
            print(f"An error occurred reading {file_path}: {e}")
            continue # Try the next file even if this one fails

    # Combine and Save
    print("Combining extracted data...")
    frames = []
    for cat, bucket_list in data_buckets.items():
        if bucket_list:
            frames.append(pd.concat(bucket_list))
    
    if not frames:
        print("No data found in any files!")
        return

    balanced_df = pd.concat(frames)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal Balanced Distribution:")
    print(balanced_df["category"].value_counts())
    
    balanced_df.to_csv(output_path, index=False)
    print(f"Saved balanced dataset to '{output_path}'")


# Check if toggle for balancing data is True or False
if RECREATE_BALANCED_DATA:
    create_balanced_dataset(FULL_SOURCE_FILES, BALANCED_FILE, TARGET_SAMPLES_PER_CLASS)
else:
    print(f"\nSkipping data creation. Using existing '{BALANCED_FILE}'...")



# ================================================
# STEP 1 - LEARNING CLUSTERS OFF BALANCED DATA
# ================================================

# Find clusters based on this artificially balanced dataset, makes it easier for the model to find clusters
print("\n--- PHASE 1: Learning Vocabulary (Balanced Data) ---")

df_bal = pd.read_csv(BALANCED_FILE) # read in balanced file
# creates a new dataframe containing only the features specified above, changes infinities to NaN and deletes rows that contain a NaN
X_bal = df_bal[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

# Standardize and PCA from sklearn
scaler = StandardScaler()
X_bal_scaled = scaler.fit_transform(X_bal)

pca = PCA(n_components=5, random_state=42) # compresses features into 5
X_bal_pca = pca.fit_transform(X_bal_scaled)

# Clustering using DBSCAN from sklearn
# lowering eps to 0.1 and min_samples to 7, has created a lot more clusters, was 6 with eps=0.5 and min=20
# but with the old config there was no correlation between clusters and categories, now there are.
# 28 clusters with most now having one category, isolates rare attack vectors, granular behavioural clustering
# Cluster 0 is not a problem that it contains both recon and ddos because:
# Recon scan: Sends a tiny packet to Port 80. Waits for a tiny reply. (Short duration, small bytes).
# DDoS flood: Sends a tiny packet to Port 80. Doesn't wait. Repeats. (Short duration, small bytes).
# they are fundamentally the same shape so cannot separate them, will ignore, cluster 20 is the goldmine, only theft
dbscan = DBSCAN(eps=0.1, min_samples=5, n_jobs=-1)
bal_clusters = dbscan.fit_predict(X_bal_pca)

# Have to map DBSCAN clusters into a KNN configuration
# this is because DBSCAN doesn't have a .predict() function, meaning if you bring in a previously unseen
# datapoint it cannot say what cluster it belongs to, doesn't learn mathematical boundaries
# can't just run DBSCAN on the full imbalanced dataset due to computational limitations
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_bal_pca, bal_clusters) # feed the KNN the cluster mappings from DBSCAN

# Print number of clusters found and datapoints in each, should be equal to 'TARGET_SAMPLES_PER_CLASS' but isn't (thats fine)
print(f"Vocabulary Learned: {len(np.unique(bal_clusters))} Clusters Found.")
print(pd.Series(bal_clusters).value_counts())


# Prepare the dataframe
stats = X_bal.copy()
stats["cluster"] = bal_clusters #add the DBSCANs classification (cluster mapping)
stats["category"] = df_bal.loc[X_bal.index, "category"] #add the truth (category in dataset)



def get_majority_label(cid, df_labeled):
    if cid == -1: return "Noise"
    
    cluster_data = df_labeled[df_labeled['cluster'] == cid]
    majority_category = cluster_data['category'].mode()[0]
    
    # Mapping UNSW-NB15 categories to MITRE ATT&CK Tactics
    mapping = {
        "Reconnaissance": "Recon (Slow)",
        "Fuzzers":        "Recon (Active/Fuzzing)",
        "DoS":            "Impact (DoS)",
        "Exploits":       "Initial Access (Exploit)",
        "Backdoors":      "Persistence (Backdoor)",
        "Shellcode":      "Execution (Shellcode)",
        "Worms":          "Lateral Movement (Worm)",
        "Analysis":       "Discovery (Analysis)",
        "Generic":        "Unknown/Generic",
        "Normal":         "Benign"
    }
    
    return mapping.get(majority_category, "Unknown")

# Dictionary of cluster IDs to MITRE mappings
cluster_to_mitre = {}
# Get all unique cluster IDs and sort them (e.g., -1, 0, 1, 2...)
unique_clusters = sorted(stats["cluster"].unique())

for cid in unique_clusters:
    label = get_majority_label(cid, stats)
    cluster_to_mitre[cid] = label
    print(f"Cluster {cid}: {label}")


# Data preperation for HMM, it can't handle the -1 produced from the DBSCAN to represent noise
# so maps each integer up 1 effectively
# Moved this up as phase 2 is now skippable
le = LabelEncoder()
# Fit on the BALANCED clusters to ensure we know all possible states
le.fit(np.unique(bal_clusters)) # fitted on bal_clusters to ensure it knows about every attack state




# ==============================================
# STEP 2 - LEARN A MODEL OFF IMBALANCED DATA
# ==============================================

print("\n--- PHASE 2: Learning Grammar (Imbalanced Data) ---")

# Read every file in the list and store them
IMBALANCED_FILE_MAIN = []
for file_path in IMBALANCED_FILES:
    print(f"Reading {file_path}...")
    imbalanced_temp = pd.read_csv(file_path, low_memory=False, names=RAW_COLUMN_NAMES, header=None)
    IMBALANCED_FILE_MAIN.append(imbalanced_temp)

# Stack them into one giant dataframe
df_imbal = pd.concat(IMBALANCED_FILE_MAIN, ignore_index=True)


# Standardize the column name
if 'attack_cat' in df_imbal.columns:
    df_imbal.rename(columns={'attack_cat': 'category'}, inplace=True)

# Clean up UNSW-NB15 whitespace and fill NaNs for normal traffic
df_imbal['category'] = df_imbal.apply(
    lambda row: 'Normal' if pd.isna(row['category']) and row['Label'] == 0 else str(row['category']).strip(), 
    axis=1
)

# Clean the combined data
# (Select features, remove Infinity, remove NaNs)
X_imbal = df_imbal[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

print(f"Loaded Imbalanced Data: {X_imbal.shape}")

# Project into Balanced Space, use the same Scaler and PCA from step 1
X_imbal_scaled = scaler.transform(X_imbal)
X_imbal_pca = pca.transform(X_imbal_scaled) # compresses features down into 5 again

df_imbal = df_imbal.loc[X_imbal.index].copy() # handle the misalignment from NaN rows again

# Assign Labels using KNN
imbal_clusters = knn.predict(X_imbal_pca) # uses KNN representation to map imbalanced dataset to the clusters
df_imbal["cluster"] = imbal_clusters

print("Real-World Cluster Distribution (Should be skewed):")
print(pd.Series(imbal_clusters).value_counts())




# ================================
# STEP 2.5 - TRAINING THE HMM
# ================================

import numpy as np
import pandas as pd
from scipy.special import softmax

class SupervisedMarkovChain:
    def __init__(self):
        # RENAMED to match hmmlearn standard (fixes your Phase 3 error)
        self.transmat_ = None 
        self.states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.state_feature_stats = {} 

    def fit(self, df, cluster_col='cluster', label_col='category', feature_cols=None):
        print("Building Composite States (Cluster + Label)...")
        
        # 1. Create Composite Column
        df['composite_state'] = df[cluster_col].astype(str) + "_" + df[label_col].astype(str)
        
        # 2. Map unique states
        unique_states = sorted(df['composite_state'].unique())
        self.states = unique_states
        self.state_to_idx = {state: i for i, state in enumerate(unique_states)}
        self.idx_to_state = {i: state for i, state in enumerate(unique_states)}
        n_states = len(unique_states)
        
        print(f"Identified {n_states} unique composite states.")

        #3. Count Transitions
        trans_counts = np.ones((n_states, n_states)) * 1e-6 # Laplace smoothing
        
        grouped = df.sort_values("stime").groupby("srcip")
        
        for srcip, group in grouped:  # Changed 'saddr' to 'srcip' here too for clarity
            seq = group['composite_state'].values
            if len(seq) < 2: continue
            
            indices = [self.state_to_idx[s] for s in seq]
            for t in range(len(indices) - 1):
                trans_counts[indices[t], indices[t+1]] += 1

        # 4. Normalize to Probabilities (transmat_)
        self.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        
        # 5. Robust Feature Profiling (Fixes the Divide by Zero / Covariance error)
        if feature_cols:
            print("Calculating robust feature profiles...")
            for state in unique_states:
                subset = df[df['composite_state'] == state][feature_cols]
                
                # Handle 'Singleton' States (only 1 packet found)
                if len(subset) > 1:
                    mean_vec = subset.mean().values
                    # Add tiny noise to diagonal to prevent singular matrix errors later
                    cov_mx = subset.cov().values + (np.eye(len(feature_cols)) * 1e-6)
                else:
                    # If only 1 packet exists, variance is zero. 
                    # We set a tiny 'synthetic' variance so we can still sample from it.
                    mean_vec = subset.iloc[0].values
                    cov_mx = np.eye(len(feature_cols)) * 1e-6
                
                self.state_feature_stats[state] = {
                    'mean': mean_vec,
                    'cov': cov_mx
                }
        
        print("Supervised Markov Chain Trained Successfully.")

    def apply_temperature(self, temperature=1.0):
        # Work on self.transmat_
        log_probs = np.log(self.transmat_ + 1e-9)
        scaled_logits = log_probs / temperature
        self.transmat_ = softmax(scaled_logits, axis=1)
        print(f"Transition matrix scaled with Temperature={temperature}")

    def generate_sequence(self, start_state, length=10):
        if start_state not in self.state_to_idx:
            raise ValueError(f"Start state '{start_state}' not found in model. Available: {self.states[:5]}...")
                
        current_idx = self.state_to_idx[start_state]
        sequence = [start_state]
            
        for _ in range(length - 1):
            probs = self.transmat_[current_idx]
            next_idx = np.random.choice(len(self.states), p=probs)
            sequence.append(self.idx_to_state[next_idx])
            current_idx = next_idx
                
        return sequence

# ==========================================
# NEW: HELPER FUNCTIONS FOR GENERATION
# ==========================================

def print_available_composites(model):
    """
    Prints all available composite states grouped by their category 
    so you can easily copy-paste them into your target list.
    """
    print("\n--- AVAILABLE COMPOSITE STATES ---") # might want to make this filter under 5 samples out or something
    
    # Group by the text part (Label) for easier reading
    organized = {}
    for state in model.states:
        # Assuming format "ClusterID_Label"
        parts = state.split('_')
        label = parts[-1] 
        if label not in organized:
            organized[label] = []
        organized[label].append(state)
        
    for label, states in organized.items():
        print(f"\n[{label.upper()}]:")
        # Print in chunks of 5 to keep it tidy
        for i in range(0, len(states), 5):
            print(f"  {states[i:i+5]}")
    print("\n----------------------------------")

# 1. The Helper Function for Temperature
def get_scaled_transition_matrix(trans_mat, T=1.0):
    """
    Returns a new transition matrix with temperature scaling applied.
    T < 1.0: Sharpening (Makes likely transitions even MORE likely).
    T > 1.0: Flattening (Increases randomness/exploration).
    """
    if T == 1.0:
        return trans_mat.copy()
    
    # Use power method to scale probabilities while preserving 0.0 constraints
    # (We don't want to create transitions that are impossible)
    scaled = np.power(trans_mat, 1.0 / T)
    
    # Re-normalize rows so they sum to 1.0
    row_sums = scaled.sum(axis=1, keepdims=True)
    # Avoid division by zero if a row is all zeros (dead end)
    row_sums[row_sums == 0] = 1.0
    
    return scaled / row_sums


import pandas as pd
import numpy as np
import time
import warnings


def robust_sample(mean, cov):
    """Safely samples from 'flat' covariance matrices."""
    epsilon = 1e-6
    cov_reg = cov + np.eye(len(mean)) * epsilon
    cov_reg = (cov_reg + cov_reg.T) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return np.random.multivariate_normal(mean, cov_reg)
        except ValueError:
            return mean
    

def generate_full_schema_traffic(model, df_original, target_states, samples_per_state=10, 
                                 seq_len=20, temperature=1.0, output_file="synthetic_unsw_nb15.csv"):

    print(f"\nStarting Generation for UNSW-NB15 (Temperature={temperature})...")
    
    # 1. EXACT UNSW-NB15 COLUMN ORDER (49 Columns)
    final_columns = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", 
        "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", 
        "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", 
        "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", 
        "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", 
        "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", 
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", 
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "category", "Label"
    ]

    all_rows = []
    
    # Temperature scaling logic
    original_transmat = model.transmat_.copy()
    if temperature != 1.0:
        model.transmat_ = get_scaled_transition_matrix(original_transmat, T=temperature)

    try:
        for start_node in target_states:
            if start_node not in model.state_to_idx: 
                print(f"Skipping {start_node} - Not found in model.")
                continue

            template_pool = df_original[df_original['composite_state'] == start_node]
            if template_pool.empty: continue
            
            print(f"  -> Generating {samples_per_state} flows for {start_node}...")
            
            for i in range(samples_per_state):
                try:
                    seq_states = model.generate_sequence(start_state=start_node, length=seq_len)
                except: continue

                # Base time for the generated sequence
                current_time = int(time.time())

                for t, state in enumerate(seq_states):
                    stats = model.state_feature_stats.get(state)
                    if not stats: continue

                    # Define sampling pool
                    current_pool = df_original[df_original['composite_state'] == state]
                    sampling_pool = template_pool if current_pool.empty else current_pool
                    template_row = sampling_pool.sample(1).iloc[0].to_dict()

                    # Categorical Sampling (UNSW just uses strings, no numbers needed!)
                    gen_proto = np.random.choice(sampling_pool['proto'].dropna().values)
                    gen_state = np.random.choice(sampling_pool['state'].dropna().values)
                    gen_service = np.random.choice(sampling_pool['service'].dropna().values)

                    # ROBUST HMM SAMPLING (Samples your 13 FEATURES)
                    feats = robust_sample(stats['mean'], stats['cov'])
                    feats = np.maximum(feats, 0) # No negative values allowed in networking

                    # Map the generated 1D array back to your FEATURES list
                    # Order: ["dur", "Spkts", "Dpkts", "sbytes", "dbytes", "Sload", "Dload", 
                    #         "smeansz", "dmeansz", "Sjit", "Djit", "Sintpkt", "Dintpkt"]
                    gen_dur = feats[0]
                    gen_Spkts = max(1, feats[1]) # Min 1 packet
                    gen_Dpkts = feats[2]
                    gen_sbytes = max(60, feats[3]) # Min 60 bytes (Ethernet limit)
                    gen_dbytes = feats[4]
                    
                    # Update the Template Row with Generated Physics
                    template_row['dur'] = gen_dur
                    template_row['Spkts'] = gen_Spkts
                    template_row['Dpkts'] = gen_Dpkts
                    template_row['sbytes'] = gen_sbytes
                    template_row['dbytes'] = gen_dbytes
                    template_row['Sload'] = feats[5]
                    template_row['Dload'] = feats[6]
                    template_row['smeansz'] = feats[7]
                    template_row['dmeansz'] = feats[8]
                    template_row['Sjit'] = feats[9]
                    template_row['Djit'] = feats[10]
                    template_row['Sintpkt'] = feats[11]
                    template_row['Dintpkt'] = feats[12]

                    # Scale dependent packet loss variables based on generated packet count
                    original_spkts = max(1, template_row.get('Spkts', 1))
                    scaling_factor = gen_Spkts / original_spkts
                    template_row['sloss'] = template_row.get('sloss', 0) * scaling_factor
                    template_row['dloss'] = template_row.get('dloss', 0) * scaling_factor

                    # Overwrite Categorical Fields
                    template_row['proto'] = gen_proto
                    template_row['state'] = gen_state
                    template_row['service'] = gen_service

                    # Time Physics Enforcement (UNSW uses Stime and Ltime)
                    template_row['Stime'] = current_time + (t * 0.5) # packets arrive chronologically
                    template_row['Ltime'] = template_row['Stime'] + gen_dur # Last time = Start time + duration

                    all_rows.append(template_row)
    finally:
        # Reset the transition matrix temperature
        if temperature != 1.0:
            model.transmat_ = original_transmat

    if all_rows:
        df_gen = pd.DataFrame(all_rows)
        
        # Ensure column order and fill missing with 0
        for col in final_columns:
            if col not in df_gen.columns:
                df_gen[col] = 0
        df_gen = df_gen[final_columns]
        
        # --- UNSW-NB15 SPECIFIC INTEGER ENFORCEMENT ---
        int_cols = [
            "sport", "dsport", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", 
            "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", 
            "trans_depth", "res_bdy_len", "is_sm_ips_ports", "ct_state_ttl", 
            "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", 
            "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", 
            "ct_dst_sport_ltm", "ct_dst_src_ltm", "Label"
        ]
        
        for col in int_cols:
            if col in df_gen.columns:
                df_gen[col] = pd.to_numeric(df_gen[col], errors='coerce')
                # 2. Fill any NaNs (the former dashes/spaces) with 0
                df_gen[col] = df_gen[col].fillna(0)
                # 3. Now it is mathematically safe to round and convert to integer
                df_gen[col] = df_gen[col].round().astype(int)

        df_gen.to_csv(output_file, index=False)
        print(f"\nSUCCESS: Generated {len(df_gen)} rows.")
        print(f"Columns reordered to match UNSW-NB15 format.")
        print(f"Saved to: {output_file}")
        return df_gen
    else:
        print("FAILED: No data generated.")
        return None

# ==========================================
# FIX: SANITIZE DATA TYPES BEFORE FITTING
# ==========================================

# 1. Define your features list explicitly if you haven't already
# (Make sure these match exactly what you want to generate)

print("Checking data types BEFORE fix:")
print(df_imbal[FEATURES].dtypes)

# 2. Force conversion to numeric (coercing errors to NaN)
for col in FEATURES:
    # This turns strings like "100" into 100.0
    # It turns junk like "Infinity" or "Text" into NaN
    df_imbal[col] = pd.to_numeric(df_imbal[col], errors='coerce')

# 3. Fill the NaNs we just created (important!)
# If 'dload' was Infinity, we replace it with the max valid value or 0
df_imbal.fillna(0, inplace=True)

print("\nChecking data types AFTER fix (Should all be float/int):")
print(df_imbal[FEATURES].dtypes)


# ==========================================
# EXECUTION BLOCK
# ==========================================

# 1. Train (Assuming df_imbal and FEATURES are already defined)
hmm_supervised = SupervisedMarkovChain()
hmm_supervised.fit(df_imbal, cluster_col='cluster', label_col='category', feature_cols=FEATURES)

# 2. VIEW what composites exist (so you can choose)
print_available_composites(hmm_supervised)

# 3. DEFINE your wish list
# You can copy-paste these from the print output above
my_target_states = [
    '4_Exploits',    
    '2_Reconnaissance', 
    '0_Normal'
]


# Generate Standard Traffic (T=1.0)
# This mimics the training data exactly.
df_std = generate_full_schema_traffic(
    model=hmm_supervised, 
    target_states=my_target_states,
    df_original=df_imbal,
    samples_per_state=100,
    seq_len=20,
    temperature=1.0,
    output_file="synthetic_traffic_standard.csv"
)

# Generate "Zero-Day" Traffic (T=1.3)
# This allows the model to take "rarer" paths, simulating slightly different attack variants.
df_wild = generate_full_schema_traffic(
    model=hmm_supervised, 
    target_states=my_target_states,
    df_original=df_imbal,
    samples_per_state=100,
    seq_len=20,
    temperature=1.3,
    output_file="synthetic_traffic_zeroday.csv"
)

# 5. Quick Peek
if df_wild is not None:
    print(df_wild.head())


def inspect_transition_probabilities(model, state_name):
    """
    Prints the top 5 most likely next states for a given state.
    """
    if state_name not in model.state_to_idx:
        print(f"State {state_name} not found.")
        return

    # Get the row of probabilities for this state
    idx = model.state_to_idx[state_name]
    probs = model.transmat_[idx]
    
    # Sort indices by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    
    print(f"\n--- TRANSITIONS FROM {state_name} ---")
    print(f"Total packets observed in training: {probs.sum() if probs.sum() > 1 else 'Normalized'}")
    
    # Print top 5 destinations
    for i in range(5):
        next_idx = sorted_indices[i]
        prob = probs[next_idx]
        if prob > 0.001: # Only show > 0.1% chance
            next_state = model.idx_to_state[next_idx]
            print(f"  -> {next_state}: {prob:.2%} chance")


inspect_transition_probabilities(hmm_supervised, '4_Exploits')