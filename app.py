"""
Customer Segmentation Web Application Backend

This Flask application provides REST API endpoints for customer segmentation
using various clustering algorithms (K-means, Hierarchical, DBSCAN).
"""

import os
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Utility function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """
    Recursively converts NumPy types in an object to native Python types.
    Works with dictionaries, lists, and individual values.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__, static_folder='static', template_folder='templates')
# Configure Flask to use the custom JSON encoder
app.json_encoder = NumpyEncoder
CORS(app)

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize global variables
df = None
scaled_data = None
pca_data = None
cluster_labels = None

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process the customer data file."""
    global df, scaled_data
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith(('.csv', '.xlsx', '.xls')):
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load the data
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Return basic data info - convert NumPy types to Python native types
            data_info = {
                'columns': df.columns.tolist(),
                'rows': int(len(df)),  # Explicitly convert to Python int
                'missing_values': int(df.isnull().sum().sum()),  # Explicitly convert to Python int
                'preview': convert_numpy_types(df.head(5).to_dict(orient='records'))  # Convert all nested NumPy types
            }
            
            return jsonify({'success': True, 'data_info': data_info})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'File format not supported'}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess the data for clustering."""
    global df, scaled_data
    
    if df is None:
        return jsonify({'error': 'Please upload data first'}), 400
    
    data = request.json
    selected_features = data.get('features', [])
    
    if not selected_features:
        return jsonify({'error': 'No features selected'}), 400
    
    try:
        # Select only numeric features from the selected ones
        numeric_features = []
        for feature in selected_features:
            if np.issubdtype(df[feature].dtype, np.number):
                numeric_features.append(feature)
            else:
                # Try to convert non-numeric features
                try:
                    df[feature] = pd.to_numeric(df[feature])
                    numeric_features.append(feature)
                except:
                    pass
        
        if not numeric_features:
            return jsonify({'error': 'No numeric features available'}), 400
        
        # Handle missing values
        df_selected = df[numeric_features].copy()
        df_selected.fillna(df_selected.mean(), inplace=True)
        
        # Scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_selected)
        
        # Generate statistics - convert NumPy types to Python native types
        stats = convert_numpy_types({
            'feature_stats': {
                feature: {
                    'mean': df_selected[feature].mean(),
                    'std': df_selected[feature].std(),
                    'min': df_selected[feature].min(),
                    'max': df_selected[feature].max()
                }
                for feature in numeric_features
            }
        })
        
        return jsonify({
            'success': True, 
            'stats': stats,
            'features_used': numeric_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/cluster', methods=['POST'])
def perform_clustering():
    """Perform clustering on the preprocessed data."""
    global df, scaled_data, pca_data, cluster_labels
    
    if scaled_data is None:
        return jsonify({'error': 'Please preprocess data first'}), 400
    
    data = request.json
    algorithm = data.get('algorithm', 'kmeans')
    params = data.get('params', {})
    
    try:
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        # Perform clustering
        if algorithm == 'kmeans':
            n_clusters = params.get('n_clusters', 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(scaled_data)
            
            # Calculate silhouette score
            silhouette = silhouette_score(scaled_data, cluster_labels) if n_clusters > 1 else 0
            
            # Get cluster centers
            centers = model.cluster_centers_
            centers_pca = pca.transform(centers)
            
            result = {
                'algorithm': 'K-Means',
                'n_clusters': int(n_clusters),
                'silhouette_score': float(silhouette),
                'clusters': [int(label) for label in cluster_labels],
                'centers': centers_pca.tolist()
            }
        
        elif algorithm == 'hierarchical':
            n_clusters = params.get('n_clusters', 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(scaled_data)
            
            # Calculate silhouette score
            silhouette = silhouette_score(scaled_data, cluster_labels) if n_clusters > 1 else 0
            
            result = {
                'algorithm': 'Hierarchical Clustering',
                'n_clusters': int(n_clusters),
                'silhouette_score': float(silhouette),
                'clusters': [int(label) for label in cluster_labels]
            }
            
        elif algorithm == 'dbscan':
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = model.fit_predict(scaled_data)
            
            # Calculate number of clusters and noise points
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            # Calculate silhouette score if there's more than one cluster and no noise
            if n_clusters > 1 and n_noise == 0:
                silhouette = silhouette_score(scaled_data, cluster_labels)
            else:
                silhouette = 0
            
            result = {
                'algorithm': 'DBSCAN',
                'n_clusters': int(n_clusters),
                'n_noise': int(n_noise),
                'silhouette_score': float(silhouette),
                'clusters': [int(label) for label in cluster_labels]
            }
        
        # Add the clustered data to the result
        df['Cluster'] = cluster_labels
        
        # Generate the visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Customer Segments using {result["algorithm"]}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # If K-means, plot the centroids
        if algorithm == 'kmeans':
            plt.scatter(
                centers_pca[:, 0], centers_pca[:, 1],
                s=200, marker='X', c='red', label='Centroids'
            )
            plt.legend()
        
        # Save the plot to a byte buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Add the plot to the result
        result['plot'] = plot_data
        
        # Cluster profiles (mean values for each feature by cluster)
        if df is not None:
            try:
                # Select only numeric columns for clustering profiles
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                # Remove the Cluster column from numerics if it's there
                if 'Cluster' in numeric_cols:
                    numeric_cols.remove('Cluster')
                
                # Add back the Cluster column for grouping
                if numeric_cols:
                    # Calculate means only for numeric columns
                    profiles_df = df[numeric_cols + ['Cluster']].groupby('Cluster').mean()
                    # Convert to dictionary and handle NumPy types
                    cluster_profiles = convert_numpy_types(profiles_df.to_dict())
                    result['cluster_profiles'] = cluster_profiles
            except Exception as profile_error:
                # Log the error but continue without cluster profiles
                print(f"Error creating cluster profiles: {profile_error}")
                # We don't add cluster_profiles to the result in this case
        
        return jsonify({'success': True, 'result': convert_numpy_types(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/export', methods=['GET'])
def export_results():
    """Export the results to a CSV file."""
    global df
    
    if df is None or 'Cluster' not in df.columns:
        return jsonify({'error': 'No clustering results to export'}), 400
    
    try:
        # Create a temporary file
        temp_file = io.BytesIO()
        df.to_csv(temp_file, index=False)
        temp_file.seek(0)
        
        return send_file(
            temp_file,
            as_attachment=True,
            download_name='customer_segments.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/optimal_k', methods=['POST'])
def find_optimal_k():
    """Find the optimal number of clusters using the Elbow method."""
    global scaled_data
    
    if scaled_data is None:
        return jsonify({'error': 'Please preprocess data first'}), 400
    
    try:
        # Calculate inertia (sum of squared distances) for different k values
        max_k = min(10, len(scaled_data) - 1)  # Don't try more clusters than data points
        k_values = range(2, max_k + 1)
        inertia_values = []
        silhouette_values = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertia_values.append(float(kmeans.inertia_))  # Convert to Python float
            silhouette_values.append(float(silhouette_score(scaled_data, kmeans.predict(scaled_data))))  # Convert to Python float
        
        # Generate the elbow plot
        plt.figure(figsize=(12, 5))
        
        # Inertia plot
        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertia_values, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        
        # Silhouette plot
        plt.subplot(1, 2, 2)
        plt.plot(k_values, silhouette_values, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method for Optimal k')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot to a byte buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        result = {
            'k_values': list(k_values),
            'inertia_values': inertia_values,
            'silhouette_values': silhouette_values,
            'plot': plot_data
        }
        
        return jsonify({'success': True, 'result': convert_numpy_types(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate_weka', methods=['GET'])
def generate_weka_file():
    """Generate a Weka ARFF file from the dataset."""
    global df
    
    if df is None:
        return jsonify({'error': 'Please upload data first'}), 400
    
    try:
        # Create a temporary file for ARFF
        temp_file = io.StringIO()
        
        # Write ARFF header
        temp_file.write("@RELATION customer_segmentation\n\n")
        
        # Write attribute information
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                temp_file.write(f"@ATTRIBUTE {column} NUMERIC\n")
            else:
                # Get unique values for categorical attributes
                unique_values = df[column].dropna().unique()
                values_str = "{" + ",".join([str(val).replace(',', '_').replace(' ', '_') for val in unique_values]) + "}"
                temp_file.write(f"@ATTRIBUTE {column} {values_str}\n")
        
        # Write data
        temp_file.write("\n@DATA\n")
        for _, row in df.iterrows():
            data_row = []
            for val in row:
                if pd.isnull(val):
                    data_row.append('?')
                elif isinstance(val, (int, float)):
                    data_row.append(str(val))
                else:
                    data_row.append(f"'{str(val).replace(',', '_').replace(' ', '_')}'")
            temp_file.write(",".join(data_row) + "\n")
        
        # Convert to BytesIO for sending
        temp_file_bytes = io.BytesIO(temp_file.getvalue().encode())
        temp_file_bytes.seek(0)
        
        return send_file(
            temp_file_bytes,
            as_attachment=True,
            download_name='customer_segments.arff',
            mimetype='text/plain'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)