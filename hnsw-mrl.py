import numpy as np
import faiss

# Parameters
d_full = 2048  # Full vector dimensionality
d_reduced_list = [16, 64, 256, 1024, 2048]  # List of reduced dimensions
coarse_neighbors = 10  # Number of neighbors to retrieve in the coarse search
top_k = 1  # Retrieve top-1 neighbor
index_path_template = "hnsw_index_d{d}.faiss"  # File name template for indices

# Load data
train_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_train_mrl1_e0_rr2048-X.npy")  # Training vectors
train_y = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_train_mrl1_e0_rr2048-y.npy")  # Labels for training vectors
val_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-X.npy")  # Query vectors
val_y = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-y.npy")  # Labels for query vectors

# Create a dictionary to store top-1 accuracy for each d
accuracy_results = {}

# Step 1: Iterate over reduced dimensions
for d_reduced in d_reduced_list:
    print(f"Building reduced-dimensional HNSW index for d = {d_reduced}...")

    # Step 2: Build reduced-dimensional index
    train_X_reduced = train_X[:, :d_reduced]
    val_X_reduced = val_X[:, :d_reduced]

    hnsw_index_reduced = faiss.IndexHNSWFlat(d_reduced, 32)
    hnsw_index_reduced.hnsw.efConstruction = 200  # Higher value improves index quality
    hnsw_index_reduced.add(train_X_reduced.astype('float32'))

    # Save the reduced-dimensional index
    index_file = index_path_template.format(d=d_reduced)
    faiss.write_index(hnsw_index_reduced, index_file)
    print(f"Saved reduced-dimensional index to {index_file}")

    # Step 3: Query coarse candidates
    print("Performing coarse search...")
    _, coarse_candidates = hnsw_index_reduced.search(val_X_reduced.astype('float32'), coarse_neighbors)

    # Step 4: Fine-grained search using full-dimensional vectors
    print("Refining search using full-dimensional vectors...")
    correct_predictions = 0

    for i, query in enumerate(val_X):
        # Get coarse candidate vectors and their labels
        candidate_ids = coarse_candidates[i]
        candidate_vectors = train_X[candidate_ids]
        candidate_labels = train_y[candidate_ids]

        # Perform exact distance calculation on coarse candidates
        distances = np.linalg.norm(candidate_vectors - query, axis=1)
        best_candidate_index = np.argmin(distances)  # Find the closest vector
        best_label = candidate_labels[best_candidate_index]  # Get its label

        # Check if the predicted label matches the ground truth
        if best_label == val_y[i]:
            correct_predictions += 1

    # Step 5: Calculate and store top-1 accuracy
    top1_accuracy = correct_predictions / len(val_y)
    accuracy_results[d_reduced] = top1_accuracy
    print(f"Top-1 Accuracy for d = {d_reduced}: {top1_accuracy:.4f}")

# Step 6: Print summary of results
print("\nSummary of Top-1 Accuracy:")
for d, accuracy in accuracy_results.items():
    print(f"d = {d}: Top-1 Accuracy = {accuracy:.4f}")