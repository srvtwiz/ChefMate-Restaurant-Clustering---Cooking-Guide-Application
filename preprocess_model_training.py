# preprocessing
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class CuisineClusteringModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df_encoded = None
        self.mlb = MultiLabelBinarizer()
        self.pca = None
        self.kmeans = None

    def preprocess_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.csv_path)
        df['cuisines_list'] = df['cuisines'].apply(lambda x: [c.strip() for c in x.split(',')])
        cuisines_encoded = self.mlb.fit_transform(df['cuisines_list'])
        cuisines_encoded_df = pd.DataFrame(cuisines_encoded, columns=self.mlb.classes_)
        self.df_encoded = pd.concat([df, cuisines_encoded_df], axis=1)
        return self.df_encoded

    def perform_pca(self, variance_threshold=0.99):
        # Dimensionality reduction with PCA
        cuisines_encoded = self.df_encoded[self.mlb.classes_]
        self.pca = PCA(n_components=variance_threshold, random_state=42)
        cuisines_pca = self.pca.fit_transform(cuisines_encoded)
        return cuisines_pca

    def train_kmeans(self, cuisines_pca, n_clusters):
        # Train KMeans clustering model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df_encoded['cluster'] = self.kmeans.fit_predict(cuisines_pca)
        return self.df_encoded

    def recommend_restaurants(self, preferred_cuisines, top_n=5):
        # Recommend restaurants based on the clustering model
        try:
            encoded_user_cuisines = self.mlb.transform([preferred_cuisines])
            user_pca = self.pca.transform(encoded_user_cuisines)
            closest_cluster = self.kmeans.predict(user_pca)[0]
            recommendations = self.df_encoded[
                (self.df_encoded['cluster'] == closest_cluster) &
                (self.df_encoded[preferred_cuisines].sum(axis=1) > 0)
            ]
            if recommendations.empty:
                return "No recommendations found for the given input."
            return recommendations[['name', 'cuisines', 'rating', 'location','average_cost_for_two','price_range','has_online_delivery','has_table_booking','is_delivering_now','currency','city']].head(top_n)
        except Exception as e:
            return f"Error in recommendations: {str(e)}"
