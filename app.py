import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Title
st.title("Interactive KMeans Clustering with Iris Dataset")

# Sidebar for user input
st.sidebar.title("Clustering Parameters")
optimal_k = st.sidebar.slider("Select the number of clusters (K)", 1, 10, 3)

# Train KMeans model and display elbow plot
def train_model():
    st.write("### Elbow Method: Optimal Number of Clusters")
    iris = datasets.load_iris()
    data = iris.data
    np.random.seed(42)

    # Define range for K values
    K_range = range(1, 11)
    ssd = []

    # Fit KMeans for each K
    for K in K_range:
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(data)
        ssd.append(kmeans.inertia_)

    # Plot elbow method
    fig, ax = plt.subplots()
    ax.plot(K_range, ssd, 'bo-', markersize=5)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Sum of Squared Distances (SSD)")
    ax.set_title("Elbow Plot")
    st.pyplot(fig)

    # Train model with user-selected K
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_optimal.fit(data)
    return kmeans_optimal, data

# Train the model
kmeans_model, iris_data = train_model()

# Visualize clustering
st.write(f"### Clustering Results with K={optimal_k}")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(iris_data)
labels = kmeans_model.labels_

fig, ax = plt.subplots()
scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50)
ax.set_title("Iris Data Clustering")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
st.pyplot(fig)

# Display cluster centers
st.write("### Cluster Centers")
st.dataframe(kmeans_model.cluster_centers_)
