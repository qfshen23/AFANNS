import numpy as np
import faiss
import math

# Parameters
coarse_neighbors = 10  # Number of neighbors to retrieve in the coarse search
top_k = 1  # Retrieve top-1 neighbor
d_full = 2048  # Full vector dimensionality
d_reduced_list = [16, 64, 256, 1024, 2048]  # List of reduced dimensions
index_path_template = "ivf_index_d{d}_cluster{clusters}.faiss"  # File name template for indices

# Load data
train_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_train_mrl1_e0_rr2048-X.npy")  # Training vectors
train_y = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_train_mrl1_e0_rr2048-y.npy")  # Labels for training vectors
val_X = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-X.npy")  # Query vectors
val_y = np.load("/home/qfshen/workspace/adanns/output/resnet50/1K_val_mrl1_e0_rr2048-y.npy")  # Labels for query vectors

# Create a dictionary to store top-1 accuracy for each d
accuracy_results = {}

# Iterate over reduced dimensions
for d_reduced in d_reduced_list:
    print(f"Building and testing IVF index for d = {d_reduced}...")

    # Step 1: Prepare reduced-dimensional data
    train_X_reduced = train_X[:, :d_reduced]
    val_X_reduced = val_X[:, :d_reduced]

    # Step 2: Compute the number of clusters (sqrt of the number of vectors)
    n_vectors = train_X_reduced.shape[0]
    n_clusters = int(math.sqrt(n_vectors))
    print(f"Number of clusters: {n_clusters}")

    # Step 3: Build and train IVF index
    quantizer = faiss.IndexFlatL2(d_reduced)  # Coarse quantizer
    ivf_index = faiss.IndexIVFFlat(quantizer, d_reduced, n_clusters)  # IVF index
    ivf_index.nprobe = 1  # Set nprobe for search

    print("Training IVF index...")
    ivf_index.train(train_X_reduced.astype('float32'))  # Train on the base vectors
    ivf_index.add(train_X_reduced.astype('float32'))  # Add base vectors to the index

    # Save the IVF index
    index_file = index_path_template.format(d=d_reduced, clusters=n_clusters)
    faiss.write_index(ivf_index, index_file)
    print(f"Saved IVF index to {index_file}")

    # Step 4: Perform validation queries
    print("Performing validation queries...")
    _, coarse_candidates = ivf_index.search(val_X_reduced.astype('float32'), coarse_neighbors)

    # Step 5: Fine-grained search and calculate top-1 accuracy
    print("Refining search using reduced-dimensional vectors...")
    correct_predictions = 0

    for i, query in enumerate(val_X_reduced):
        # Get coarse candidate vectors and their labels
        candidate_ids = coarse_candidates[i]
        candidate_vectors = train_X_reduced[candidate_ids]
        candidate_labels = train_y[candidate_ids]

        # Perform exact distance calculation on coarse candidates
        distances = np.linalg.norm(candidate_vectors - query, axis=1)
        best_candidate_index = np.argmin(distances)  # Find the closest vector
        best_label = candidate_labels[best_candidate_index]  # Get its label

        # Check if the predicted label matches the ground truth
        if best_label == val_y[i]:
            correct_predictions += 1

    # Calculate top-1 accuracy
    top1_accuracy = correct_predictions / len(val_y)
    accuracy_results[d_reduced] = top1_accuracy
    print(f"Top-1 Accuracy for d = {d_reduced}: {top1_accuracy:.4f}")

# Step 6: Print summary of results
print("\nSummary of Top-1 Accuracy for IVF:")
for d, accuracy in accuracy_results.items():
    print(f"d = {d}: Top-1 Accuracy = {accuracy:.4f}")