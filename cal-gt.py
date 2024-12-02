import numpy as np
import faiss

train_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_train_mrl1_e0_rr2048-X.npy")  # Training vectors
val_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-X.npy")  # Query vectors

# Parameters
d_list = [16, 64, 256, 1024, 2048]  # List of dimensions to compute ground truth
top_k = 10  # Number of nearest neighbors to compute for ground truth
groundtruth_file_template = "groundtruth_d{d}.npy"  # Template for ground truth file names

# Iterate over each dimension
for d in d_list:
    print(f"\nComputing ground truth for dimension d = {d}...")

    # Reduce dimensionality of train and validation sets
    train_X_reduced = train_X[:, :d]
    val_X_reduced = val_X[:, :d]

    # Build brute-force index for the current dimension
    print(f"Building brute-force index for d = {d}...")
    flat_index = faiss.IndexFlatL2(d)  # Exact L2 distance search
    flat_index.add(train_X_reduced.astype('float32'))

    # Perform search for ground truth
    print(f"Performing search for top-{top_k} neighbors...")
    _, groundtruth = flat_index.search(val_X_reduced.astype('float32'), top_k)

    # Save ground truth to file
    groundtruth_file = groundtruth_file_template.format(d=d)
    np.save(groundtruth_file, groundtruth)
    print(f"Ground truth saved to {groundtruth_file}")
