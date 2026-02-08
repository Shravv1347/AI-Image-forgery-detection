"""
Distance Based Models Module
Implements: K-Nearest Neighbors, K-Means Clustering, Hierarchical Clustering, DBSCAN
(From Unit 2.3 of syllabus)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns


class KNNClassifier:
    """
    K-Nearest Neighbors Classification
    (From Unit 2.3: Neighbors and Examples, Nearest Neighbors Classification)
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train KNN classifier"""
        print(f"\nTraining K-Nearest Neighbors (k={self.n_neighbors})...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("KNN training completed!")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def find_optimal_k(self, X_train, y_train, X_val, y_val, k_range=range(1, 21)):
        """Find optimal k value using validation set"""
        accuracies = []
        
        print("\nFinding optimal k value...")
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, weights=self.weights, metric=self.metric)
            knn.fit(X_train, y_train)
            acc = knn.score(X_val, y_val)
            accuracies.append(acc)
            print(f"k={k}: Validation Accuracy = {acc:.4f}")
        
        optimal_k = k_range[np.argmax(accuracies)]
        print(f"\nOptimal k = {optimal_k} with accuracy = {max(accuracies):.4f}")
        
        # Plot k vs accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Neighbors (k)', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('KNN: Finding Optimal k', fontsize=14)
        plt.grid(alpha=0.3)
        plt.axvline(x=optimal_k, color='r', linestyle='--', 
                    label=f'Optimal k={optimal_k}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/claude/knn_optimal_k.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return optimal_k, accuracies


class KMeansClustering:
    """
    K-Means Clustering Algorithm
    (From Unit 2.3: Distance based clustering-K means Algorithm)
    """
    
    def __init__(self, n_clusters=2, max_iter=300, n_init=10):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=42
        )
        self.labels_ = None
        self.cluster_centers_ = None
    
    def fit(self, X):
        """Fit K-Means clustering"""
        print(f"\nFitting K-Means with {self.n_clusters} clusters...")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        
        # Calculate clustering metrics
        silhouette = silhouette_score(X, self.labels_)
        inertia = self.model.inertia_
        
        print(f"K-Means clustering completed!")
        print(f"Inertia (within-cluster sum of squares): {inertia:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        
        return self.labels_
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.model.predict(X)
    
    def find_optimal_clusters(self, X, k_range=range(2, 11)):
        """
        Find optimal number of clusters using Elbow Method and Silhouette Score
        """
        inertias = []
        silhouette_scores = []
        
        print("\nFinding optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:  # Silhouette score requires at least 2 clusters
                sil_score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(sil_score)
                print(f"k={k}: Inertia={kmeans.inertia_:.4f}, Silhouette={sil_score:.4f}")
        
        # Plot elbow curve and silhouette scores
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow curve
        axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14)
        axes[0].grid(alpha=0.3)
        
        # Silhouette scores
        axes[1].plot(list(k_range)[1:], silhouette_scores, marker='o', 
                     linewidth=2, markersize=8, color='green')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/kmeans_optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        optimal_k = list(k_range)[1:][np.argmax(silhouette_scores)] + 1
        print(f"\nRecommended k = {optimal_k} (based on Silhouette Score)")
        
        return optimal_k, inertias, silhouette_scores


class HierarchicalClusteringModel:
    """
    Hierarchical Clustering
    (From Unit 2.3: Hierarchical clustering)
    """
    
    def __init__(self, n_clusters=2, linkage='ward'):
        """
        linkage: 'ward', 'complete', 'average', 'single'
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        self.labels_ = None
    
    def fit(self, X):
        """Fit hierarchical clustering"""
        print(f"\nFitting Hierarchical Clustering ({self.linkage} linkage)...")
        self.labels_ = self.model.fit_predict(X)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(X, self.labels_)
        
        print(f"Hierarchical clustering completed!")
        print(f"Silhouette Score: {silhouette:.4f}")
        
        return self.labels_
    
    def plot_dendrogram(self, X, truncate_mode='lastp', p=30):
        """Plot dendrogram for hierarchical clustering"""
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        print("\nGenerating dendrogram...")
        linkage_matrix = linkage(X, method=self.linkage)
        
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)', 
                  fontsize=14)
        plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        plt.savefig('/home/claude/hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Dendrogram saved as hierarchical_dendrogram.png")


class DBSCANModel:
    """
    DBSCAN - Density-Based Spatial Clustering of Applications with Noise
    (From Unit 2.3 of syllabus)
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        self.labels_ = None
        self.n_clusters_ = None
        self.n_noise_ = None
    
    def fit(self, X):
        """Fit DBSCAN clustering"""
        print(f"\nFitting DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        self.labels_ = self.model.fit_predict(X)
        
        # Number of clusters (excluding noise points labeled as -1)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_noise_ = list(self.labels_).count(-1)
        
        print(f"DBSCAN clustering completed!")
        print(f"Number of clusters: {self.n_clusters_}")
        print(f"Number of noise points: {self.n_noise_}")
        
        # Calculate silhouette score (only if we have at least 2 clusters)
        if self.n_clusters_ >= 2:
            # Exclude noise points for silhouette calculation
            mask = self.labels_ != -1
            if np.sum(mask) > 0:
                silhouette = silhouette_score(X[mask], self.labels_[mask])
                print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
        
        return self.labels_
    
    def find_optimal_eps(self, X, k=5):
        """
        Find optimal eps using k-distance graph
        """
        from sklearn.neighbors import NearestNeighbors
        
        print(f"\nFinding optimal eps using {k}-distance graph...")
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)
        
        # Sort distances
        distances = np.sort(distances[:, -1], axis=0)
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel('Data Points sorted by distance', fontsize=12)
        plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
        plt.title('K-distance Graph for DBSCAN eps Selection', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/claude/dbscan_eps_selection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("K-distance graph saved. Look for the 'elbow' point to select eps.")
        
        # Suggest an eps value (using 95th percentile as rough estimate)
        suggested_eps = np.percentile(distances, 95)
        print(f"Suggested eps value: {suggested_eps:.4f}")
        
        return suggested_eps
