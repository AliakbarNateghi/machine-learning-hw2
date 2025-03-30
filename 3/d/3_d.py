import math

# Training data: Each entry is ((x1, x2), label)
training_data = [
    ((1, 1), 0),
    ((2, 2), 0),
    ((3, 4), 1),
    ((4, 3), 1),
]

# Test point
test_point = (2.5, 2.5)

def euclidean_distance(p, q):
    """Compute Euclidean distance between 2D points p and q."""
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def knn_predict(test_pt, data, k):
    """
    Classify test_pt using k-Nearest Neighbors.
    data is a list of ((x1, x2), label) tuples.
    """
    # 1. Compute distances to all points
    distances = []
    for (features, label) in data:
        dist = euclidean_distance(test_pt, features)
        distances.append((dist, label))
    
    # 2. Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # 3. Pick the k nearest neighbors
    k_nearest = distances[:k]
    
    # 4. Count majority label
    #    Extract labels of these k points and pick the majority
    labels = [label for (_, label) in k_nearest]
    # majority vote
    predicted_label = max(set(labels), key=labels.count)
    
    return predicted_label

# Test the function for k=1 and k=3
k1_prediction = knn_predict(test_point, training_data, k=1)
k3_prediction = knn_predict(test_point, training_data, k=3)

print("Prediction for k=1:", k1_prediction)
print("Prediction for k=3:", k3_prediction)
