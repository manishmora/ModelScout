# ModelScout
ModelScout is a Django-based web application that allows users to upload datasets and train both supervised and unsupervised machine learning models with ease. It provides detailed performance metrics, visualizations (confusion matrices, ROC curves, clustering plots), and model comparisons — all in a user-friendly web interface.

# 🧠 ModelScout

**ModelScout** is a Django-based web application that empowers users to train both **Supervised** and **Unsupervised** Machine Learning models on custom datasets, with interactive visualizations, comparison tables, and performance metrics. Built for learning, experimentation, and rapid prototyping.

---

## 🚀 Features

### ✅ Supervised Learning
Train and compare:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- AdaBoost
- Decision Tree
- Naive Bayes

Visualizations & Metrics:
- 📊 Confusion Matrix
- 📉 ROC Curve (for binary classification)
- 🔍 Feature Importances
- 📈 SHAP Values (tree-based models)
- 🧪 Classification Report
- 🎯 Accuracy, Cross-Validation Scores, Training Time

---

### 🔍 Unsupervised Learning
Analyze structure in unlabeled datasets using:
- KMeans Clustering
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Models

Visualizations & Scores:
- 🌐 Cluster Plots
- 📏 Silhouette Scores
- 📊 Dimensionality Reduction (PCA, t-SNE)
- 📌 Label overlays and color-coded clusters

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/modelscout.git
cd modelscout

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver
