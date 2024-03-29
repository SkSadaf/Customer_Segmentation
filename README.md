
<h3><b>Customer Segmentation Web App</b></h3>


<ins>**Overview**</ins>

This web application allows users to perform customer segmentation on their datasets efficiently. It includes features such as user registration, login, and the ability to upload CSV files for clustering. The application leverages clustering algorithms like DBSCAN, Agglomerative Clustering, and K-Means to categorize customers based on their behavioral attributes.


<ins>**Technologies Used**</ins>

**Flask:** Python web framework for building the application

**SQLAlchemy:** SQL toolkit for Python, used for database management

**Scikit-learn:** Machine learning library for implementing clustering algorithms

**Pandas:** Data manipulation and analysis library for handling datasets

**Matplotlib:** Data visualization library for generating clustering plots

**Bcrypt:** Hashing library for secure password storage

**HTML/CSS:** Front-end design and user interface

**Bootstrap:** Front-end framework for responsive and mobile-first web development


<ins>**Features**</ins>

**User Authentication:** Users can register and log in securely to access the segmentation features.

**CSV Upload:** Users can upload datasets in CSV format for customer segmentation.

**Clustering Algorithms:** The application employs DBSCAN, Agglomerative Clustering, and K-Means algorithms to perform efficient customer segmentation.

**Interactive Visualization:** Users can visualize clustering results through interactive plots generated by Matplotlib.

**Cluster-wise Data:** After segmentation, users can download individual CSV files for each customer segment.


<ins>**Requirements**</ins>

Python 3.6 or higher

Flask, SQLAlchemy, Scikit-learn, Pandas, Matplotlib, Bcrypt


<ins>**Additional Information**</ins>

The application uses SQLite for database management, and the database file is named database.db.

Uploaded files are stored in the static/uploads directory, and cluster data is saved in the static/cluster_data directory.
