# K-Means Clustering

![K-Means Clustering](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)
## Problem Statement

K-Means Clustering is an unsupervised machine learning algorithm used to partition a dataset into groups or clusters based on their similarity. The goal is to divide the data points into k distinct clusters in such a way that the within-cluster variation is minimized and the between-cluster variation is maximized.

## Mathematics

The mathematical formulation of K-Means Clustering involves the following elements:

- **Data Points**: Let X = {x₁, x₂, ..., xₙ} be the set of n data points that need to be clustered.

- **Cluster Centers**: Let C = {c₁, c₂, ..., cₖ} be the set of k cluster centers, where each cᵢ represents the centroid of a cluster.

- **Cluster Assignment**: Each data point xⱼ is assigned to one of the k clusters based on its similarity to the cluster centers. This assignment is represented by the set S = {s₁, s₂, ..., sₙ}, where sⱼ represents the cluster label of xⱼ.

- **Objective Function**: The objective of K-Means Clustering is to minimize the following objective function:

  ![function](CodeCogsEqn.png)

  Here, ‖ · ‖ represents the Euclidean distance between a data point and a cluster center.

## Algorithm

The K-Means Clustering algorithm consists of the following steps:

1. **Initialization**: Randomly initialize k cluster centers.

2. **Assignment**: Assign each data point to the nearest cluster center based on the Euclidean distance.

3. **Update**: Recalculate the cluster centers by taking the mean of all the data points assigned to each cluster.

4. **Repeat**: Repeat steps 2 and 3 until convergence. Convergence occurs when the cluster assignments no longer change or a maximum number of iterations is reached.

5. **Output**: The algorithm outputs the final cluster centers and the cluster assignments of all the data points.

## Conclusion

K-Means Clustering is a popular algorithm used for partitioning data points into distinct clusters. It is widely used in various applications, such as image segmentation, customer segmentation, and anomaly detection. By understanding the problem statement, the mathematical formulation, and the algorithm, you can apply K-Means Clustering to your own datasets and uncover meaningful patterns and insights.
