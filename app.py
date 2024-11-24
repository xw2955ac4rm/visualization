import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import chardet

# Tabs for the app
tabs = st.tabs(["Learn About KMeans", "Try Demo Data", "Try with Your Data"])

# Tab 1: Learn About KMeans
with tabs[0]:
    st.title("Learn About KMeans Clustering")
    st.write("""
    KMeans clustering is an unsupervised machine learning algorithm that groups data into a pre-defined number of clusters (K). 
    It works by minimizing the within-cluster variance and is widely used for data segmentation and pattern recognition tasks.
    """)
    st.markdown("""
    ### Resources to Learn More
    - [Scikit-Learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - [Comprehensive Guide on KMeans Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/)
    - [KMeans Algorithm on Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
    """)

# Tab 2: Try Demo Data
with tabs[1]:
    st.title("Try KMeans with Demo Data (Iris Dataset)")

    def download_demo_data():
        iris = datasets.load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['target'] = iris.target

        # Provide a download link for the dataset
        st.write("### Download Demo Data (Iris Dataset)")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Iris Dataset as CSV",
            data=csv,
            file_name='iris_demo_data.csv',
            mime='text/csv',
        )

        return data

    # Provide download option
    demo_data = download_demo_data()

    # Display clustering demo with Iris dataset
    def train_with_demo_data(data):
        np.random.seed(42)

        # Define the range for K values
        K_range = range(1, 11)
        ssd = []

        # Fit KMeans for each K
        for K in K_range:
            kmeans = KMeans(n_clusters=K, random_state=42)
            kmeans.fit(data)
            ssd.append(kmeans.inertia_)

        # Elbow Plot
        st.write("### Elbow Method to Determine Optimal K")
        fig, ax = plt.subplots()
        ax.plot(K_range, ssd, 'bo-', markersize=5)
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Sum of Squared Distances (SSD)")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # Train KMeans with the optimal K
        optimal_K = 3
        kmeans_optimal = KMeans(n_clusters=optimal_K, random_state=42)
        kmeans_optimal.fit(data)

        # Visualize Clusters
        st.write("### Cluster Visualization")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        labels = kmeans_optimal.labels_

        fig, ax = plt.subplots()
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50)
        ax.set_title("Clusters Visualized (Iris Data)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)

    train_with_demo_data(demo_data.iloc[:, :-1])

# Tab 3: Try with Your Data
with tabs[2]:
    st.title("Try KMeans with Your Own Data")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file with numeric data", type=["csv"])

    if uploaded_file:
        try:
            # Detect file encoding
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

            # Reset file pointer and read with detected encoding
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding=encoding)

            # Validate if all columns are numeric
            if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                st.error("All features must be numeric. Please check your data.")
            else:
                # Display the uploaded data
                st.write("### Data Preview")
                st.dataframe(data.head())

                # Train the KMeans model and visualize results
                def train_with_uploaded_data(data):
                    st.write("### Elbow Method to Determine Optimal K")
                    np.random.seed(42)

                    # Define the range for K values
                    K_range = range(1, 11)
                    ssd = []

                    # Fit KMeans for each K
                    for K in K_range:
                        kmeans = KMeans(n_clusters=K, random_state=42)
                        kmeans.fit(data)
                        ssd.append(kmeans.inertia_)

                    # Elbow Plot
                    fig, ax = plt.subplots()
                    ax.plot(K_range, ssd, 'bo-', markersize=5)
                    ax.set_xlabel("Number of Clusters (K)")
                    ax.set_ylabel("Sum of Squared Distances (SSD)")
                    ax.set_title("Elbow Method")
                    st.pyplot(fig)

                    # Train KMeans with the optimal K
                    optimal_K = st.slider("Select the number of clusters (K)", 1, 10, 3)
                    kmeans_optimal = KMeans(n_clusters=optimal_K, random_state=42)
                    kmeans_optimal.fit(data)

                    # Visualize Clusters
                    st.write("### Cluster Visualization")
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(data)
                    labels = kmeans_optimal.labels_

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50)
                    ax.set_title("Clusters Visualized (Uploaded Data)")
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    st.pyplot(fig)

                    # Display Cluster Centers
                    st.write("### Cluster Centers")
                    st.dataframe(pd.DataFrame(kmeans_optimal.cluster_centers_, columns=data.columns))

                train_with_uploaded_data(data)
        except UnicodeDecodeError:
            st.error("Please save the file as UTF-8 and try again.")
        except Exception as e:
            st.error(f"An error occurred while processing your data: {e}")
    else:
        st.write("Upload your CSV file to get started.")
