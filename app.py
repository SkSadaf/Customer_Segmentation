from flask import Flask, request, render_template, redirect, session, url_for, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import pandas as pd
import os
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from werkzeug.utils import secure_filename 
from uuid import uuid4

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
app.config['SQLALCHEMY_ECHO'] = True


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()
    print("Database tables created")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # Handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Generate unique ID
        new_id = uuid4()

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error='User already exists')

        # Create and store new user in the database
        new_user = User(email=email, password=password, name=name)
        db.session.add(new_user)
        db.session.commit()

        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

UPLOAD_FOLDER = 'static/uploads'
CLUSTER_DATA_FOLDER = 'static/cluster_data'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the 'static/uploads' directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def perform_clustering(data, attribute1, attribute2):
    # Extract relevant features for clustering
    X = data[[attribute1, attribute2]]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=4)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    # Hierarchical Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    agg_labels = agg_clustering.fit_predict(X_scaled)

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Create a directory to store cluster data
    if not os.path.exists(CLUSTER_DATA_FOLDER):
        os.makedirs(CLUSTER_DATA_FOLDER)

    # Save the data of each cluster to separate CSV files
    for cluster_num in range(num_clusters):
        cluster_data = data[dbscan_labels == cluster_num]
        cluster_data.to_csv(f'{CLUSTER_DATA_FOLDER}/cluster_{cluster_num}.csv', index=False)

    # Return clustering labels for visualization
    cluster_data = {}
    for cluster_num in range(num_clusters):
        cluster_data[cluster_num] = data[dbscan_labels == cluster_num]

    clustering_results = {
        'dbscan_labels': dbscan_labels,
        'agg_labels': agg_labels,
        'kmeans_labels': kmeans_labels,
        'cluster_data': cluster_data,
        'num_clusters': num_clusters
    }

    return clustering_results


# Generate and save plots as image files
def generate_plots(data, dbscan_labels, agg_labels, kmeans_labels, attribute1, attribute2):
    # Calculate silhouette scores for each clustering algorithm
    from sklearn.metrics import silhouette_score

    dbscan_silhouette_score = silhouette_score(data, dbscan_labels)
    agg_silhouette_score = silhouette_score(data, agg_labels)
    kmeans_silhouette_score = silhouette_score(data, kmeans_labels)

    # Generate and save plots with silhouette scores in titles
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(data[attribute1], data[attribute2], c=dbscan_labels, cmap='rainbow', s=10)
    plt.title(f'DBSCAN Clustering (Silhouette Score: {dbscan_silhouette_score:.3f})', fontsize=8)
    plt.xlabel(attribute1)
    plt.ylabel(attribute2)
    plt.savefig('static/plot_dbscan.png')

    plt.subplot(132)
    plt.scatter(data[attribute1], data[attribute2], c=agg_labels, cmap='rainbow', s=10)
    plt.title(f'Hierarchical Agglomerative Clustering (Silhouette Score: {agg_silhouette_score:.3f})', fontsize=8)
    plt.xlabel(attribute1)
    plt.ylabel(attribute2)
    plt.savefig('static/plot_agg.png')

    plt.subplot(133)
    plt.scatter(data[attribute1], data[attribute2], c=kmeans_labels, cmap='rainbow', s=10)
    plt.title(f'K-Means Clustering (Silhouette Score: {kmeans_silhouette_score:.3f})', fontsize=8)
    plt.xlabel(attribute1)
    plt.ylabel(attribute2)
    plt.savefig('static/plot_kmeans.png')

@app.route('/dashboard')
def upload_file():
    if session.get('email'):
        user = User.query.filter_by(email=session['email']).first()
        return render_template('upload.html', user=user)
    return redirect('/login')

@app.route('/segment', methods=['POST'])
def segment_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file with its original name
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Load the dataset
        data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        attribute1 = request.form['attribute1']
        attribute2 = request.form['attribute2']

        # Perform clustering
        clustering_results = perform_clustering(data, attribute1, attribute2)
        dbscan_labels = clustering_results['dbscan_labels']
        agg_labels = clustering_results['agg_labels']
        kmeans_labels = clustering_results['kmeans_labels']

        # Generate and save plots
        generate_plots(data, dbscan_labels, agg_labels, kmeans_labels, attribute1, attribute2)

        # Generate and save cluster-wise CSV files
        for cluster_num, cluster_df in clustering_results['cluster_data'].items():
            cluster_df.to_csv(os.path.join(CLUSTER_DATA_FOLDER, f'cluster_{cluster_num}.csv'), index=False)

        # Generate download links for cluster-wise data
        num_clusters = clustering_results['num_clusters']  # Get the actual number of clusters
        download_links = [
            (cluster_num, f'/download/cluster_{cluster_num}.csv')
            for cluster_num in range(num_clusters)
        ]

        return render_template('results.html', download_links=download_links)


@app.route('/results')
def results():
    # Load the dataset and attributes
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'Mall_Customers.csv'))
    attribute1 = request.form['attribute1']
    attribute2 = request.form['attribute2']

    # Retrieve clustering results
    clustering_results = perform_clustering(data, attribute1, attribute2)
    num_clusters = clustering_results['num_clusters']
    
    download_links = [
        (cluster_num, f'/download/cluster_{cluster_num}.csv')
        for cluster_num in range(num_clusters)
    ]
    
    return render_template('results.html', download_links=download_links)



@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(CLUSTER_DATA_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)