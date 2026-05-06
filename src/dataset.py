"""
Dataset partitioning module for federated learning.

Provides multiple data partitioning strategies to simulate different
federated learning scenarios with varying levels of data heterogeneity.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
from scipy.stats import wasserstein_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def partition_iid(df: pd.DataFrame, n_clients: int = 5, seed: int = 42) -> List[pd.DataFrame]:
    """
    Randomly shuffle and split df into n_clients equal parts.
    
    IID (Independent and Identically Distributed) partitioning assumes
    each client has data drawn from the same distribution. This is the
    least heterogeneous scenario.
    
    Args:
        df: Input dataframe with 'Label' column
        n_clients: Number of clients to partition data into
        seed: Random seed for reproducibility
        
    Returns:
        List of dataframes, one per client
    """
    np.random.seed(seed)
    
    # Randomly shuffle indices
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    # Split into n_clients equal parts
    client_dfs = []
    samples_per_client = len(df) // n_clients
    
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < n_clients - 1 else len(df)
        
        client_indices = indices[start_idx:end_idx]
        client_dfs.append(df.iloc[client_indices].reset_index(drop=True))
        
        logger.info(f"Client {i}: {len(client_dfs[i])} samples "
                   f"(pos: {(client_dfs[i]['Label'] == 1).sum()}, "
                   f"neg: {(client_dfs[i]['Label'] == 0).sum()})")
    
    logger.info(f"IID partition created: {n_clients} clients")
    return client_dfs


def partition_by_day(df: pd.DataFrame, n_clients: int = 5) -> List[pd.DataFrame]:
    """
    Split df by day column. Each client gets one day.
    If n_clients < 8, group remaining days into last client.
    
    This partitioning simulates scenarios where clients correspond to
    different days of network traffic, which naturally have different
    distributions of attack types.
    
    Args:
        df: Input dataframe with 'Timestamp' column
        n_clients: Number of clients to partition into
        
    Returns:
        List of dataframes, one per client
    """
    # Extract day from timestamp
    df = df.copy()
    if 'Timestamp' in df.columns:
        df['Day'] = pd.to_datetime(df['Timestamp']).dt.date
    else:
        # If no Timestamp, create one based on index
        logger.warning("No 'Timestamp' column found. Creating synthetic days...")
        days_per_chunk = len(df) // 8  # Assume 8 days total
        df['Day'] = (df.index // days_per_chunk).astype(str)
    
    unique_days = sorted(df['Day'].unique())
    logger.info(f"Found {len(unique_days)} unique days: {unique_days}")
    
    client_dfs = []
    
    if n_clients >= len(unique_days):
        # One client per day
        for day in unique_days:
            day_df = df[df['Day'] == day].drop(columns=['Day']).reset_index(drop=True)
            client_dfs.append(day_df)
            logger.info(f"Day {day}: {len(day_df)} samples "
                       f"(pos: {(day_df['Label'] == 1).sum()}, "
                       f"neg: {(day_df['Label'] == 0).sum()})")
    else:
        # Group days into n_clients
        days_per_client = len(unique_days) / n_clients
        
        for i in range(n_clients):
            start_day_idx = int(i * days_per_client)
            end_day_idx = int((i + 1) * days_per_client) if i < n_clients - 1 else len(unique_days)
            
            days_for_client = unique_days[start_day_idx:end_day_idx]
            client_df = df[df['Day'].isin(days_for_client)].drop(columns=['Day']).reset_index(drop=True)
            client_dfs.append(client_df)
            logger.info(f"Client {i} (days {days_for_client}): {len(client_df)} samples "
                       f"(pos: {(client_df['Label'] == 1).sum()}, "
                       f"neg: {(client_df['Label'] == 0).sum()})")
    
    logger.info(f"Day-based partition created: {len(client_dfs)} clients")
    return client_dfs


def partition_by_attack_family(df: pd.DataFrame, n_clients: int = 5, 
                               seed: int = 42) -> List[pd.DataFrame]:
    """
    Assign attack families to clients such that each client gets 1-2 attack types
    plus a proportional share of normal traffic.
    
    This partitioning creates highly heterogeneous non-IID scenarios where
    different clients specialize in detecting different types of attacks:
    - Client 0: DDoS + Normal
    - Client 1: Brute Force + Normal
    - Client 2: Web Attacks + Normal
    - Client 3: Botnet + Normal
    - Client 4: Infiltration + Heartbleed + remaining rare classes + Normal
    
    Args:
        df: Input dataframe with attack type labels
        n_clients: Number of clients (default 5 recommended)
        seed: Random seed for reproducibility
        
    Returns:
        List of dataframes, one per client, each with different attack types
    """
    np.random.seed(seed)
    df = df.copy()
    
    # Identify attack types/families in the Label column
    # Assuming Label column contains attack type names or we extract from another column
    unique_labels = df['Label'].unique()
    logger.info(f"Unique labels: {unique_labels}")
    
    # Define attack families based on typical CICIDS2017 attacks
    # This mapping assumes labels contain attack type information
    attack_families = {
        'DDoS': ['DDoS', 'DDoS-ACK', 'DDoS-UDP', 'DDoS-PSHACK', 'DDoS-SYN'],
        'Brute Force': ['SSH-Brute Force', 'FTP-Brute Force', 'Brute Force'],
        'Web Attacks': ['SQL Injection', 'XSS', 'Web Attack', 'Infiltration'],
        'Botnet': ['Botnet', 'Bot'],
        'Other': ['Heartbleed', 'Infiltration', 'Port Scan']
    }
    
    # Map each label to a family
    label_to_family = {}
    for label in unique_labels:
        label_str = str(label).lower()
        found = False
        for family, keywords in attack_families.items():
            if any(kw.lower() in label_str for kw in keywords):
                label_to_family[label] = family
                found = True
                break
        if not found:
            # Default: if 'normal' or similar, treat as normal; otherwise as 'Other'
            if label_str in ['normal', 'benign', 'legitimate', 0]:
                label_to_family[label] = 'Normal'
            else:
                label_to_family[label] = 'Other'
    
    logger.info(f"Label to family mapping: {label_to_family}")
    
    # Add family column
    df['Family'] = df['Label'].map(label_to_family)
    
    # Get normal traffic
    normal_df = df[df['Family'] == 'Normal'].reset_index(drop=True)
    attack_df = df[df['Family'] != 'Normal'].reset_index(drop=True)
    
    logger.info(f"Normal samples: {len(normal_df)}")
    logger.info(f"Attack samples: {len(attack_df)}")
    
    # Define attack assignments to clients
    attack_assignments = {
        0: ['DDoS'],
        1: ['Brute Force'],
        2: ['Web Attacks'],
        3: ['Botnet'],
        4: ['Other']
    }
    
    # If n_clients != 5, adjust assignments
    if n_clients < 5:
        # Merge attack families
        attack_assignments = {
            i: list(attack_families.keys())[i::n_clients] 
            for i in range(n_clients)
        }
    elif n_clients > 5:
        # Split some families across clients
        logger.warning(f"n_clients={n_clients} > 5. Distributing attacks across clients...")
        attack_assignments = {i: [] for i in range(n_clients)}
        attack_families_list = list(attack_families.keys())
        for i, family in enumerate(attack_families_list):
            attack_assignments[i % n_clients].append(family)
    
    # Distribute normal traffic proportionally
    samples_per_client = len(normal_df) / n_clients
    
    # Partition normal traffic
    normal_indices = np.arange(len(normal_df))
    np.random.shuffle(normal_indices)
    
    client_dfs = []
    for client_id in range(n_clients):
        # Get assigned attack families
        assigned_families = attack_assignments.get(client_id, ['Other'])
        
        # Get attacks for this client
        client_attacks = attack_df[
            attack_df['Family'].isin(assigned_families)
        ].reset_index(drop=True)
        
        # Get normal traffic for this client
        start_idx = int(client_id * samples_per_client)
        end_idx = int((client_id + 1) * samples_per_client) if client_id < n_clients - 1 else len(normal_df)
        client_normal_indices = normal_indices[start_idx:end_idx]
        client_normal = normal_df.iloc[client_normal_indices].reset_index(drop=True)
        
        # Combine
        client_data = pd.concat([client_attacks, client_normal], ignore_index=True)
        client_data = client_data.drop(columns=['Family']).reset_index(drop=True)
        client_dfs.append(client_data)
        
        logger.info(f"Client {client_id} (families: {assigned_families}): {len(client_data)} samples "
                   f"(attacks: {len(client_attacks)}, normal: {len(client_normal)})")
    
    logger.info(f"Attack family partition created: {n_clients} clients with diverse attack types")
    return client_dfs


def compute_emd(client_dfs: List[pd.DataFrame]) -> np.ndarray:
    """
    Compute Earth Mover Distance between each pair of client label distributions
    to quantify heterogeneity.
    
    EMD measures the minimum cost to transform one distribution into another.
    Higher EMD values indicate more different label distributions across clients
    (more heterogeneous/non-IID).
    
    Args:
        client_dfs: List of dataframes, one per client
        
    Returns:
        NxN distance matrix where N is number of clients
    """
    n_clients = len(client_dfs)
    emd_matrix = np.zeros((n_clients, n_clients))
    
    # Get label distributions for each client
    client_distributions = []
    for i, client_df in enumerate(client_dfs):
        # Compute label distribution (proportion of each label)
        label_counts = client_df['Label'].value_counts(normalize=True)
        client_distributions.append(label_counts)
        logger.info(f"Client {i} label distribution: {dict(label_counts)}")
    
    # Compute pairwise EMD
    unique_labels = sorted(set(label for dist in client_distributions for label in dist.index))
    
    for i in range(n_clients):
        for j in range(i, n_clients):
            # Get distributions aligned to all unique labels
            dist_i = np.array([client_distributions[i].get(label, 0) for label in unique_labels])
            dist_j = np.array([client_distributions[j].get(label, 0) for label in unique_labels])
            
            # Normalize to ensure they sum to 1
            dist_i = dist_i / (dist_i.sum() + 1e-10)
            dist_j = dist_j / (dist_j.sum() + 1e-10)
            
            # Compute Wasserstein distance (1-D EMD)
            emd_value = wasserstein_distance(dist_i, dist_j)
            emd_matrix[i, j] = emd_value
            emd_matrix[j, i] = emd_value
    
    logger.info(f"EMD Matrix:\n{emd_matrix}")
    return emd_matrix


def analyze_partitions(client_dfs: List[pd.DataFrame], partition_name: str = ""):
    """
    Analyze and print statistics about the partitions.
    
    Args:
        client_dfs: List of client dataframes
        partition_name: Name of partition scheme for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Partition Analysis: {partition_name}")
    logger.info(f"{'='*60}")
    
    total_samples = sum(len(df) for df in client_dfs)
    total_positive = sum((df['Label'] == 1).sum() for df in client_dfs)
    total_negative = sum((df['Label'] == 0).sum() for df in client_dfs)
    
    logger.info(f"Total clients: {len(client_dfs)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total positive: {total_positive} ({total_positive/total_samples*100:.2f}%)")
    logger.info(f"Total negative: {total_negative} ({total_negative/total_samples*100:.2f}%)")
    
    # Compute imbalance per client
    imbalances = []
    for i, client_df in enumerate(client_dfs):
        n_pos = (client_df['Label'] == 1).sum()
        n_neg = (client_df['Label'] == 0).sum()
        total = len(client_df)
        imbalance = n_pos / (n_neg + 1e-10) if n_neg > 0 else float('inf')
        imbalances.append(imbalance)
    
    logger.info(f"\nImbalance ratio per client (positive/negative):")
    for i, imb in enumerate(imbalances):
        logger.info(f"  Client {i}: {imb:.3f}")
    
    logger.info(f"Average imbalance: {np.mean(imbalances):.3f}")
    logger.info(f"Imbalance std: {np.std(imbalances):.3f}")
    
    # Compute EMD
    emd = compute_emd(client_dfs)
    logger.info(f"\nData Heterogeneity (EMD):")
    logger.info(f"  Mean EMD: {emd[np.triu_indices_from(emd, k=1)].mean():.3f}")
    logger.info(f"  Max EMD: {emd[np.triu_indices_from(emd, k=1)].max():.3f}")
    logger.info(f"  Min EMD: {emd[np.triu_indices_from(emd, k=1)].min():.3f}")


if __name__ == "__main__":
    # Example usage with synthetic data
    logger.info("Creating synthetic CICIDS2017-like data for testing...")
    
    n_samples = 1000
    np.random.seed(42)
    
    synthetic_data = pd.DataFrame({
        'Label': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Timestamp': pd.date_range('2017-07-03', periods=n_samples, freq='1min'),
        'Flow_Bytes/s': np.random.exponential(100, n_samples),
        'Flow_Packets/s': np.random.exponential(10, n_samples)
    })
    
    # Test IID partition
    logger.info("\n" + "="*60)
    logger.info("Testing IID Partition")
    logger.info("="*60)
    iid_clients = partition_iid(synthetic_data, n_clients=5)
    analyze_partitions(iid_clients, "IID")
    
    # Test day-based partition
    logger.info("\n" + "="*60)
    logger.info("Testing Day-Based Partition")
    logger.info("="*60)
    day_clients = partition_by_day(synthetic_data, n_clients=5)
    analyze_partitions(day_clients, "Day-Based")
    
    logger.info("\nDataset module ready for use!")
