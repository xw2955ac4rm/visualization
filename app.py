import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import chardet

# Tabs for the app
tabs = st.tabs(["Learn About KMeans", "KMeans Clustering"])

# Tab 1: Learn About KMeans
with tabs[0]:
    st.title("Learn About KMeans Clustering")
    st.write("""
    KMeans clustering is an unsupervised machine learning algorithm that groups data into a pre-defined number of clusters (K). 
    It works by minimizing the within-cluster variance and is widely used for data segmentation and pattern recognition tasks.
    """)

    # Add external learning resource links
    st.markdown("""
    ### Resources to Learn More
    - [Scikit-Learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - [Comprehensive Guide on KMeans Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/)
    - [KMeans Algorithm on Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
    """)

    # Add a button to visit an external website
    if st.button("Visit External Learning Platform"):
        st.markdown("[Click here to explore more about KMeans](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)", unsafe_allow_html=True)

# Tab 2: Clustering
with tabs[1]:
    st.title("Interactive KMeans Clustering with Custom Dataset")
    
    # Sidebar only visible in the "KMeans Clustering" tab
    with st.sidebar:
        st.title("Clustering Parameters")
        optimal_k = st.slider("Select the number of clusters (K)", 1, 10, 3)

        # File Upload
        uploaded_file = st.file_uploader("Upload a CSV file for clustering", type=["csv"])

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
                    def train_model(data):
                        st.write("### Elbow Method: Determine the Optimal Number of Clusters")
                        np.random.seed(42)

                        # Define the range of K values
                        K_range = range(1, 11)
                        ssd = []

                        # Compute Sum of Squared Distances (SSD) for each K
                        for K in K_range:
                            kmeans = KMeans(n_clusters=K, random_state=42)
                            kmeans.fit(data)
                            ssd.append(kmeans.inertia_)

                        # Plot the Elbow Curve
                        fig, ax = plt.subplots()
                        ax.plot(K_range, ssd, 'bo-', markersize=5)
                        ax.set_xlabel("Number of Clusters (K)")
                        ax.set_ylabel("Sum of Squared Distances (SSD)")
                        ax.set_title("Elbow Method")
                        st.pyplot(fig)

                        # Train KMeans with the selected K
                        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
                        kmeans_optimal.fit(data)
                        return kmeans_optimal, data

                    # Train the KMeans model
                    kmeans_model, cluster_data = train_model(data)

                    # Visualize the clustering results
                    st.write(f"### Clustering Results for K={optimal_k}")
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(cluster_data)
                    labels = kmeans_model.labels_

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50)
                    ax.set_title("Cluster Visualization")
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    st.pyplot(fig)

                    # Display Cluster Centers
                    st.write("### Cluster Centers")
                    st.dataframe(pd.DataFrame(kmeans_model.cluster_centers_, columns=data.columns))
            except UnicodeDecodeError:
                st.error("Please save the file as UTF-8 and try again.")
            except Exception as e:
                st.error(f"An error occurred while processing your data: {e}")
        else:
            st.write("Please upload a CSV file to perform clustering.")
            st.write("The dataset should consist of numeric features only, with rows representing samples and columns representing features.")

# Sidebar content hidden in the "Learn About KMeans" tab
if tabs[0]:
    st.sidebar.empty()
