import numpy as np
import faiss

# Parameters
d_list = [16, 64, 256, 1024, 2048]  # List of reduced dimensions
top_k = 10  # Number of nearest neighbors for recall calculation
nprobe = 1  # Number of clusters to search in IVF
index_path_template = "ivf_index_d{d}_cluster1131.faiss"  # Template for IVF index file
groundtruth_file = "groundtruth_d2048.npy"  # Full-dimension ground truth file

# Load full-dimension data
train_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_train_mrl1_e0_rr2048-X.npy")  # Training vectors
val_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-X.npy")  # Query vectors
groundtruth = np.load(groundtruth_file)  # Ground truth for full dimensions

# Dictionary to store recall results
recall_results_ivf = {}

# Iterate over reduced dimensions
for d in d_list:
    print(f"\nTesting IVF Recall for reduced dimension d = {d}...")

    # Reduce validation set to current dimension for search
    val_X_reduced = val_X[:, :d]

    # Load IVF index for this dimension
    ivf_index_file = index_path_template.format(d=d)
    print(f"Loading IVF index from {ivf_index_file}...")
    ivf_index = faiss.read_index(ivf_index_file)
    ivf_index.nprobe = nprobe

    # Perform search using reduced-dimension index
    print("Performing search...")
    _, ivf_neighbors = ivf_index.search(val_X_reduced.astype('float32'), top_k)

    # Compute Recall@1 and Recall@10 using full-dimension ground truth
    ivf_recall_at_1 = np.mean([groundtruth[i, 0] in ivf_neighbors[i, :1] for i in range(len(val_X))])
    ivf_recall_at_10 = np.mean([groundtruth[i, 0] in ivf_neighbors[i, :10] for i in range(len(val_X))])

    # Store results
    recall_results_ivf[d] = (ivf_recall_at_1, ivf_recall_at_10)
    print(f"Recall@1: {ivf_recall_at_1:.4f}, Recall@10: {ivf_recall_at_10:.4f}")

# Print summary of results
print("\nSummary of IVF Recall Results:")
for d, (recall1, recall10) in recall_results_ivf.items():
    print(f"d = {d}: Recall@1 = {recall1:.4f}, Recall@10 = {recall10:.4f}")