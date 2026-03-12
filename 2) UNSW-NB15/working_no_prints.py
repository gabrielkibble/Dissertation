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
FULL_SOURCE_FILES = ["UNSW_NB15_testing-set.csv",
                     "UNSW_NB15_training-set.csv"
                     ]

BALANCED_FILE = "balanced_NB15_iot.csv"                  # The output file we create/use
OUTPUT_SEQ_FILE = "novel_attack_sequences2.npy"

# Need all of it
IMBALANCED_FILES = FULL_SOURCE_FILES



# ========================
# Toggles and Parameters
# ========================

# Toggles for what code is run
RECREATE_BALANCED_DATA = False  # Set to False if "balanced_bot_iot.csv" already exists
RECREATE_SEQUENCES = False      # Set to False if "novel_attack_sequences.npy" already exists

# Parameters
TARGET_SAMPLES_PER_CLASS = 3000
SEQ_LENGTH = 20 # have a graph printed now to justify this, maybe change to a little higher
STRIDE = 10 # should be half of seq_length for overlap, if above changes, change this
TEMPERATURE = 1.2 # see google doc for justification for why its not that large


FEATURES = [
    "dur",                      # Duration
    "Spkts", "Dpkts",           # Packet counts
    "sbytes", "dbytes",         # Volume
    "Sload", "Dload",           # Speed (Bits per second)
    "smeansz", "dmeansz",       # Average packet sizes
    "Sjit", "Djit",             # Jitter (mSec)
    "Sintpkt", "Dintpkt"        # Interpacket arrival times
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
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                
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
    imbalanced_temp = pd.read_csv(file_path, low_memory=False)
    IMBALANCED_FILE_MAIN.append(imbalanced_temp)

# Stack them into one giant dataframe
df_imbal = pd.concat(IMBALANCED_FILE_MAIN, ignore_index=True)

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

        # 3. Count Transitions
        trans_counts = np.ones((n_states, n_states)) * 1e-6 # Laplace smoothing
        
        grouped = df.sort_values("stime").groupby("saddr")
        for saddr, group in grouped:
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

# Create a dictionary of 'flgs' -> 'flgs_number' from your real data
unique_flags = df_imbal[['flgs', 'flgs_number']].drop_duplicates().sort_values('flgs_number')

print("--- Your Dataset's Flag Mapping ---")
for index, row in unique_flags.iterrows():
    print(f"{row['flgs_number']} -> '{row['flgs']}'")


# Create a dictionary of 'proto' -> 'proto_number' from your real data
unique_protos = df_imbal[['proto', 'proto_number']].dropna().drop_duplicates().sort_values('proto_number')

print("--- Your Dataset's Protocol Mapping ---")
for index, row in unique_protos.iterrows():
    print(f"{row['proto_number']} -> '{row['proto']}'")


import pandas as pd
import numpy as np
import time
import warnings

   # 1. SETUP MAPPINGS (To ensure Numbers match Strings)
    # We scan the original DF to build a truth dictionary (e.g., {'INT': 4, 'FIN': 1})
    # This guarantees the numbers "match how many there are" in the real data.
    
    # Helper to build map safely
def build_map(df, str_col, num_col):
    if str_col in df.columns and num_col in df.columns:
        # Drop duplicates to get unique pairs
        pairs = df[[str_col, num_col]].dropna().drop_duplicates()
        return dict(zip(pairs[str_col], pairs[num_col]))
    return {}


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
                                 seq_len=20, temperature=1.0, output_file="synthetic_bot_iot.csv"):

    print(f"\nStarting Generation (with Smart Scaling)...")
    
    # 1. WARNING SYSTEM: Checks if 'df_original' actually has the stats columns
    required_stats = ['TnBPSrcIP', 'TnP_PerProto', 'AR_P_Proto_P_SrcIP']
    missing = [c for c in required_stats if c not in df_original.columns]
    if missing:
        print(f"⚠️ WARNING: df_original is missing columns: {missing}")
        print("   The generated stats will be 0. Please pass the FULL RAW dataset.")

    # 2. DEFINITIONS: Which columns need scaling?
    # Scale these based on Bytes
    byte_stats = ['sbytes', 'dbytes'] 
    pkt_stats = ['Spkts', 'Dpkts', 'sloss', 'dloss']
    

 
    state_map = build_map(df_original, 'state', 'state_number')
    #flgs_map = build_map(df_original, 'flgs', 'flgs_number')
    proto_map_rev = build_map(df_original, 'proto', 'proto_number') # 'udp' -> 3
# 2. Flag Map (Your Hardcoded Fix + Orphans)
    flgs_map = {
        'e': 1, 'e s': 2, 'e d': 3, 'e *': 4, 'e g': 5, 
        'eU': 6, 'e &': 7, 'e   t': 8, 'e  D': 9, 
        'e dS': 10, 'e    F': 11
    }

    proto_map_str_to_num = {
        'tcp': 1, 'arp': 2, 'udp': 3, 'icmp': 4, 'ipv6-icmp': 5
    }

    # Create Reverse Map: Number -> Name (e.g. 3 -> 'udp')
    proto_map_num_to_str = {v: k for k, v in proto_map_str_to_num.items()}

    # Standard State Map (BoT-IoT default)
    state_map = {'INT': 4, 'FIN': 1, 'CON': 2, 'REQ': 3, 'RST': 5, 'URP': 6}

    # Protocol Map (Standardizing common numbers)
    # This ensures consistency if the template has 'udp' but no number
    proto_map_rev = {'udp': 17, 'tcp': 6, 'icmp': 1, 'arp': 0, 'ipv6-icmp': 58}
    
    # 2. DEFINE EXACT COLUMN ORDER
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

    print(f"\nStarting Generation (Corrected Schema)...")
    
    try:
        for start_node in target_states:
            if start_node not in model.state_to_idx: continue

            # Filter templates for this specific composite state
            template_pool = df_original[df_original['composite_state'] == start_node]
            if template_pool.empty: continue
            
            print(f"  -> Generating {samples_per_state} flows for {start_node}...")
            
            for i in range(samples_per_state):
                try:
                    seq_states = model.generate_sequence(start_state=start_node, length=seq_len)
                except: continue

                for t, state in enumerate(seq_states):
                    stats = model.state_feature_stats.get(state)
                    if not stats: continue

                    # 1. DEFINE POOL FIRST (We need it for Protocol Sampling)
                    # We grab the real data for this state immediately
                    current_pool = df_original[df_original['composite_state'] == state]
                    
                    # Fallback Logic: If specific state is empty/rare, use the start node's pool
                    if current_pool.empty:
                        sampling_pool = template_pool
                        # Grab template now
                        template_row = template_pool.sample(1).iloc[0].to_dict()
                    else:
                        sampling_pool = current_pool
                        # Grab template now
                        template_row = current_pool.sample(1).iloc[0].to_dict()

                    # 2. CATEGORICAL SAMPLING FOR PROTOCOL
                    gen_proto_num = np.random.choice(sampling_pool['proto_number'].values)

                    # 3. ROBUST HMM SAMPLING (For Bytes & Speed)
                    mean_vec = stats['mean']
                    cov_mx = stats['cov']
                    
                    # Proactive Fix for Singular Matrix
                    cov_reg = cov_mx + np.eye(len(mean_vec)) * 1e-6 # Add tiny noise
                    cov_reg = (cov_reg + cov_reg.T) / 2             # Force symmetry
                    
                    feats = robust_sample(stats['mean'], stats['cov'])
                    feats = np.maximum(feats, 0)
                    gen_bytes = feats[0]
                    gen_dload = feats[2]

                    # --- NEW SCALING LOGIC START ---
                    
                    # 1. Calculate Multiplier (Generated vs Original)
                    original_bytes = template_row.get('bytes', 1.0)
                    if original_bytes == 0: original_bytes = 1.0 # Avoid divide by zero
                    
                    scaling_factor = gen_bytes / original_bytes
                    
                    # 2. Safety Clip (Prevent 1000x explosions)
                    scaling_factor = np.clip(scaling_factor, 0.1, 10.0)

                    # 3. Apply Multiplier to Byte Stats
                    for col in byte_stats:
                        if col in template_row:
                            template_row[col] = template_row[col] * scaling_factor

                    # 4. Apply Multiplier to Packet Stats
                    for col in pkt_stats:
                        if col in template_row:
                            template_row[col] = template_row[col] * scaling_factor
                            
                    # --- NEW SCALING LOGIC END ---

                    # --- OVERWRITE CORE FIELDS ---
                    template_row['proto_number'] = gen_proto_num
                    # ... (rest of code is same)
                    
                    # Update protocol string name using your reversed map
                    proto_name_map = {v: k for k, v in proto_map_rev.items()}
                    # If the number is new (e.g. 17), update the string to 'udp'
                    template_row['proto'] = proto_name_map.get(gen_proto_num, template_row['proto'])

                    # 2. Update Stats
                    template_row['bytes'] = gen_bytes
                    template_row['sbytes'] = gen_bytes
                    template_row['dbytes'] = 0
                    template_row['rate'] = gen_dload / 8.0 if gen_dload > 0 else 0
                    
                    # 3. Flags & State Consistency
                    curr_flgs = template_row.get('flgs', 'e')
                    curr_state = template_row.get('state', 'INT')
                    
                    template_row['flgs_number'] = flgs_map.get(curr_flgs, 1)
                    template_row['state_number'] = state_map.get(curr_state, 4) 

                    # 4. Identity & Time
                    template_row['pkSeqID'] = int(time.time() * 1000000) + t
                    template_row['stime'] = template_row['stime'] + (t * 0.1)
                    template_row['seq'] = i
                    
                    all_rows.append(template_row)
    finally:
        if temperature != 1.0:
            model.transmat_ = original_transmat

    if all_rows:
        df_gen = pd.DataFrame(all_rows)
        
        # Fill missing columns with 0 or empty string to prevent errors
        for col in final_columns:
            if col not in df_gen.columns:
                df_gen[col] = 0
        
        # Enforce the EXACT order requested
        df_gen = df_gen[final_columns]

        # ... inside generate_full_schema_traffic_final ...
        
        # --- FIX: FORCE INTEGERS ---
        # These columns can NEVER be decimals in real life
        int_cols = ['pkts', 'bytes', 'spkts', 'dpkts', 
                    'sbytes', 'dbytes', 'flgs_number', 
                    'state_number', 'proto_number', 'seq']
        
        # Ensure column order
        for col in final_columns:
            if col not in df_gen.columns: df_gen[col] = 0
        df_gen = df_gen[final_columns]
        
        df_gen.to_csv(output_file, index=False)
        
        for col in int_cols:
            if col in df_gen.columns:
                # Round to nearest whole number, then convert to Int
                df_gen[col] = df_gen[col].round().astype(int)
                
        # --- FIX: FORCE ATTACK LABEL TO INT ---
        if 'attack' in df_gen.columns:
             df_gen['attack'] = df_gen['attack'].round().astype(int)


        # 2. Force Valid Byte Sizes (The "Physics" Fix)
        # Ensures every packet has at least 60 bytes (Ethernet minimum)
        min_bytes = df_gen['pkts'] * 60
        df_gen['bytes'] = np.maximum(df_gen['bytes'], min_bytes)
        df_gen['sbytes'] = df_gen['bytes'] # Sync source bytes

        # Ensure column order
        for col in final_columns:
            if col not in df_gen.columns: df_gen[col] = 0
        df_gen = df_gen[final_columns]
        
        df_gen.to_csv(output_file, index=False)
        print(f"\nSUCCESS: Generated {len(df_gen)} rows.")
        print(f"Columns reordered to match BoT-IoT original format.")
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
    '12_Worms'
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


inspect_transition_probabilities(hmm_supervised, '')