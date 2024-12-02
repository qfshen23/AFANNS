
import numpy as np
import faiss

# Parameters
d_list = [16, 64, 256, 1024, 2048]  # List of dimensions to test
top_k = 10  # Evaluate Recall@10
index_path_template = "hnsw_index_d{d}.faiss"  # Template for HNSW index file
groundtruth_file_template = "groundtruth_d{d}.npy"  # Template for ground truth files

# Load validation data
val_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-X.npy")  # Query vectors

# Dictionary to store recall results
recall_results_hnsw = {}

# Iterate over each dimension
for d in d_list:
    print(f"\nTesting HNSW Recall for dimension d = {d}...")

    # Reduce validation set to the current dimension
    val_X_reduced = val_X[:, :d]

    # Load ground truth for this dimension
    groundtruth_file = groundtruth_file_template.format(d=d)
    print(f"Loading ground truth from {groundtruth_file}...")
    groundtruth = np.load(groundtruth_file)

    # Load HNSW index for this dimension
    hnsw_index_file = index_path_template.format(d=d)
    print(f"Loading HNSW index from {hnsw_index_file}...")
    hnsw_index = faiss.read_index(hnsw_index_file)

    # Perform search
    print("Performing search...")
    _, hnsw_neighbors = hnsw_index.search(val_X_reduced.astype('float32'), top_k)

    # Compute Recall@1 and Recall@10
    hnsw_recall_at_1 = np.mean([groundtruth[i, 0] in hnsw_neighbors[i, :1] for i in range(len(val_X))])
    hnsw_recall_at_10 = np.mean([groundtruth[i, 0] in hnsw_neighbors[i, :10] for i in range(len(val_X))])

    # Store results
    recall_results_hnsw[d] = (hnsw_recall_at_1, hnsw_recall_at_10)
    print(f"Recall@1: {hnsw_recall_at_1:.4f}, Recall@10: {hnsw_recall_at_10:.4f}")

# Print summary of results
print("\nSummary of HNSW Recall Results:")
for d, (recall1, recall10) in recall_results_hnsw.items():
    print(f"d = {d}: Recall@1 = {recall1:.4f}, Recall@10 = {recall10:.4f}")