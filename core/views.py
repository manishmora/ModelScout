from django.shortcuts import render, get_object_or_404, redirect
from .models import Dataset
from .forms import DatasetUploadForm
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
import time
import numpy as np
from sklearn.metrics import roc_curve, auc, silhouette_score, silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import warnings

# Suppressing warnings for clarity
warnings.filterwarnings("ignore")

# Define paths for saving datasets and plots
MEDIA_ROOT = os.path.join(settings.MEDIA_ROOT, 'plots')
DATASET_PATH = os.path.join(settings.MEDIA_ROOT, 'datasets')
os.makedirs(MEDIA_ROOT, exist_ok=True)

def save_plot(fig, filename):
    plot_path = os.path.join(settings.MEDIA_ROOT, 'plots', filename)
    fig.savefig(plot_path, dpi=300)  # Save with higher resolution
    plt.close(fig)
    return os.path.join('plots', filename)

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_name = form.cleaned_data['name']
            dataset_file = form.cleaned_data['file']
            Dataset.objects.create(name=dataset_name, file=dataset_file)
            return redirect('dataset_list')
    else:
        form = DatasetUploadForm()

    return render(request, 'upload.html', {'form': form})

def dataset_list(request):
    datasets = Dataset.objects.all()
    return render(request, 'dataset_list.html', {'datasets': datasets})

def delete_dataset(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    dataset.delete()
    return redirect('dataset_list')

def dataset_details(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    dataset_path = dataset.file.path

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        return render(request, 'error.html', {'message': f"File not found: {dataset_path}"})

    dataset_stats = {
        'columns': df.columns.tolist(),
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }

    if request.method == 'POST':
        return redirect('train_supervised_model', dataset_id=dataset_id)

    context = {
        'dataset': dataset,
        'dataset_stats': dataset_stats,
    }
    return render(request, 'core/dataset_details.html', context)

# Train Model View
def train_model(request, dataset_id, algo_type):  
    if algo_type not in ['supervised', 'unsupervised']:
        return HttpResponse("Error: Invalid algorithm type selected in URL.")

    dataset = get_object_or_404(Dataset, id=dataset_id)
    try:
        df = pd.read_csv(dataset.file.path)
    except FileNotFoundError:
        return render(request, 'error.html', {'message': "Dataset file not found."})

    return train_supervised_models(request, dataset, df) if algo_type == "supervised" else train_unsupervised_models(request, dataset, df)

def train_supervised_models(request, dataset, df):
    X = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    y_bin = label_binarize(y, classes=np.arange(n_classes)) if n_classes > 2 else y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    model_results = []
    all_visualizations = []

    best_accuracy = 0
    best_model_name = None

    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = round(time.time() - start_time, 3)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
            cv_scores = cross_val_score(model, X, y, cv=5)

            model_results.append({
                "model_name": name,
                "accuracy": round(accuracy, 4),
                "cv_score_mean": round(np.mean(cv_scores), 4),
                "cv_score_std": round(np.std(cv_scores), 4),
                "training_time": training_time,
                "report": report
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(f"{name} - Confusion Matrix", fontsize=16)
            cm_url = settings.MEDIA_URL + save_plot(fig_cm, f"cm_{name}_{uuid.uuid4().hex}.png")
            all_visualizations.append({"model_name": name, "type": "Confusion Matrix", "url": cm_url})

            # ROC Curve (only binary)
            if n_classes == 2 and hasattr(model, "predict_proba"):
                fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                fig_roc, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=14)
                ax.set_ylabel('True Positive Rate', fontsize=14)
                ax.set_title(f"{name} - ROC Curve", fontsize=16)
                ax.legend(loc="lower right")
                roc_url = settings.MEDIA_URL + save_plot(fig_roc, f"roc_{name}_{uuid.uuid4().hex}.png")
                all_visualizations.append({"model_name": name, "type": "ROC Curve", "url": roc_url})

            # Feature Importances
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                feature_names = X.columns
                fig_fi, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feature_importances, y=feature_names, ax=ax)
                ax.set_title(f"{name} - Feature Importances", fontsize=16)
                fi_url = settings.MEDIA_URL + save_plot(fig_fi, f"fi_{name}_{uuid.uuid4().hex}.png")
                all_visualizations.append({"model_name": name, "type": "Feature Importances", "url": fi_url})

            # SHAP
            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                fig_shap, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                shap_url = settings.MEDIA_URL + save_plot(fig_shap, f"shap_{name}_{uuid.uuid4().hex}.png")
                all_visualizations.append({"model_name": name, "type": "SHAP Summary", "url": shap_url})

        except Exception as e:
            print(f"Error occurred while training model {name}: {e}")
            continue

    best_model = max(model_results, key=lambda x: x.get("accuracy", 0), default=None)
    best_model_name = best_model.get('model_name') if best_model else None

    # Filter best model's visualizations
    best_model_visualizations = [viz for viz in all_visualizations if viz['model_name'] == best_model_name]

    return render(request, 'train_model_results.html', {
        "dataset": dataset,
        "model_results": model_results,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_model_visualizations": best_model_visualizations,
        "all_visualizations": [viz for viz in all_visualizations if viz['model_name'] != best_model_name],
        "algo_type": "Supervised Learning"
    })


def train_unsupervised_models(request, dataset, df):

    X = df.iloc[:, :-1]
    models = {
        "KMeans": KMeans(n_clusters=3, random_state=42),
        "DBSCAN": DBSCAN(),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
        "Gaussian Mixture": GaussianMixture(n_components=3),
    }

    models_info = []
    all_visualizations = []
    best_model_visualizations = []

    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X)
            training_time = round(time.time() - start_time, 3)

            y_pred = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
            silhouette_avg = silhouette_score(X, y_pred)

            models_info.append({
                "model_name": name,
                "model": model,
                "y_pred": y_pred,
                "silhouette_score": round(silhouette_avg, 4),
                "training_time": training_time
            })

        except Exception as e:
            print(f"Error occurred while training model {name}: {e}")
            continue

    model_results = [
        {
            "model_name": m["model_name"],
            "silhouette_score": m["silhouette_score"],
            "training_time": m["training_time"]
        }
        for m in models_info
    ]

    best_model = max(models_info, key=lambda x: x["silhouette_score"], default=None)
    best_model_name = best_model["model_name"] if best_model else None

    for m in models_info:
        name = m["model_name"]
        model = m["model"]
        y_pred = m["y_pred"]

        # PCA for 2D projection
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        # Cluster Plot with Legend
        fig_cluster, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_pred, palette="Set2", ax=ax, legend='full')
        ax.set_title(f"{name} - Cluster Plot")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend(title="Cluster Label")
        cluster_url = settings.MEDIA_URL + save_plot(fig_cluster, f"cluster_{name}_{uuid.uuid4().hex}.png")

        cluster_viz = {"model_name": name, "type": "Cluster Plot", "url": cluster_url}
        all_visualizations.append(cluster_viz)
        if name == best_model_name:
            best_model_visualizations.append(cluster_viz)

        # Silhouette Plot with Labels
        n_clusters = len(np.unique(y_pred))
        silhouette_vals = silhouette_samples(X, y_pred)
        y_lower = 10
        fig_sil, ax = plt.subplots(figsize=(8, 6))

        for i in range(n_clusters):
            ith_silhouette_vals = silhouette_vals[y_pred == i]
            ith_silhouette_vals.sort()
            size_cluster_i = ith_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7, label=f"Cluster {i}")
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_score(X, y_pred), color="red", linestyle="--", label="Average Score")
        ax.set_title(f"{name} - Silhouette Plot")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        ax.legend(loc="best")
        sil_url = settings.MEDIA_URL + save_plot(fig_sil, f"sil_{name}_{uuid.uuid4().hex}.png")

        sil_viz = {"model_name": name, "type": "Silhouette Score", "url": sil_url}
        all_visualizations.append(sil_viz)
        if name == best_model_name:
            best_model_visualizations.append(sil_viz)

        # Elbow Plot for KMeans only
        if isinstance(model, KMeans):
            inertia_values = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, random_state=42).fit(X)
                inertia_values.append(km.inertia_)
            fig_elbow, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(1, 11), inertia_values, marker='o', linestyle='--')
            ax.set_title(f"{name} - Elbow Method")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Inertia")
            elbow_url = settings.MEDIA_URL + save_plot(fig_elbow, f"elbow_{name}_{uuid.uuid4().hex}.png")

            elbow_viz = {"model_name": name, "type": "Elbow Method", "url": elbow_url}
            all_visualizations.append(elbow_viz)
            if name == best_model_name:
                best_model_visualizations.append(elbow_viz)

    return render(request, 'train_model_results.html', {
        "dataset": dataset,
        "model_results": model_results,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_model_visualizations": best_model_visualizations,
        "all_visualizations": [viz for viz in all_visualizations if viz['model_name'] != best_model_name],
        "algo_type": "Unsupervised Learning"
    })

def predict_user_input(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df = pd.read_csv(dataset.file.path)

    columns = df.columns[:-1]  
    prediction = None

    if request.method == 'POST':
        feature_values = [float(request.POST.get(feature, 0)) for feature in columns]  # Use default value
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        model = RandomForestClassifier()  # Make sure RandomForestClassifier is imported
        model.fit(X, y)
        prediction = model.predict([feature_values])[0]

    context = {
        'dataset': dataset,
        'features': columns,
        'prediction': prediction,
        'feature_info': df.dtypes[:-1].to_dict()  
    }
    return render(request, 'core/predict_user_input.html', context)