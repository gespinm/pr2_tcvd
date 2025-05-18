import click
import csv
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


_OUTPUT_FOLDER = "dataset/"
_OUTPUT_FILEPATH = "Medicaldataset.csv"
_OUTPUT_FILEPATH_CLEAN = "dataset_clean.csv"
_OUTPUT_PNG_FILEPATH_CLEAN = "plot_clean.png"


def _read_csv(filename):
    """Read data from a CSV file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    return pd.read_csv(filename)


def _clean(df):
    """Clean and balance df."""
    df = df.copy()
    df.replace(['nan', -1, -1.0, '', ' ', '-', '1...'], np.nan, inplace=True)

    df['Result'] = df['Result'].astype(str).str.strip().str.lower()
    df["Result"] = df["Result"].map({"positive": 1, "negative": 0})

    for col in df.columns:
        if col != "Result":
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    count_class_0 = (df["Result"] == 0).sum()
    count_class_1 = (df["Result"] == 1).sum()
    minority_count = min(count_class_0, count_class_1)
    df_class_0 = df[df["Result"] == 0].sample(n=minority_count, random_state=42)
    df_class_1 = df[df["Result"] == 1].sample(n=minority_count, random_state=42)
    df_balanced = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced


def _save_to_csv(data):
    """Save extracted data to a CSV file."""
    os.makedirs(os.path.dirname(_OUTPUT_FOLDER), exist_ok=True)
    
    data_rows = data.values.tolist()
    with open(_OUTPUT_FOLDER + _OUTPUT_FILEPATH_CLEAN, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data.columns)
        writer.writerows(data_rows)
    

def _save_to_png(data):
    """Save extracted data to a PNG file."""
    os.makedirs(os.path.dirname(_OUTPUT_FOLDER), exist_ok=True)
    counts = data["Result"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.bar(["Negative", "Positive"], counts)
    plt.savefig(_OUTPUT_FOLDER + _OUTPUT_PNG_FILEPATH_CLEAN)
    plt.close()


def _analyze(df):
    output_folder = os.path.join(os.path.dirname(__file__), "..", "model_plots")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    X = df[["Age", "Gender", "Heart rate", "Systolic blood pressure",
            "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]]
    y = df["Result"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model supervisat:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_scaled)

    cm_log = confusion_matrix(y_test, y_pred_log, labels=[0, 1])
    disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["negative", "positive"])
    plt.figure()
    disp_log.plot()
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig(os.path.join(output_folder, "SM.png"))
    plt.close()

    print("\nLogistic Regression Model")
    print("Accuracy:", log_reg.score(X_test_scaled, y_test))
    print("Confusion Matrix:\n", cm_log)


    # Model no supervisat:
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_2d = pca.transform(kmeans.cluster_centers_)

    cluster_labels = {}
    for cluster_id in [0, 1]:
        labels_in_cluster = y[clusters == cluster_id]
        majority_label = labels_in_cluster.mode()[0]
        cluster_labels[cluster_id] = majority_label

    y_pred_kmeans = np.array([cluster_labels[c] for c in clusters])
    accuracy_kmeans = accuracy_score(y, y_pred_kmeans)
    print("\n K-means Model")
    print("Accuracy:", accuracy_kmeans)  

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1])
    plt.title("KMeans - 2D PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "UM.png"))
    plt.close()



@click.command()
@click.option('--path', required=False, default=_OUTPUT_FOLDER + _OUTPUT_FILEPATH, help='Target URL to scrape')
@click.option('--save-csv', required=False, is_flag=True, default=False, help='Save locally the output in a CSV file')
@click.option('--save-png', required=False, is_flag=True, default=False, help='Save locally a plot of the output in a png file')
def main(path, save_csv:bool, save_png:bool):
    df = _read_csv(path)
    if not df.empty:
        clean_df = _clean(df)
        print("Data cleaning was successfull")
        if save_csv:
            _save_to_csv(clean_df)
            print(f"    - Results saved in csv: {_OUTPUT_FILEPATH_CLEAN}")
        if save_png:
            _save_to_png(clean_df)
            print(f"    - Results saved in png: {_OUTPUT_PNG_FILEPATH_CLEAN}")
        _analyze(clean_df)
        print("Data was succesfully processed")

    else:  
        raise Exception(f"There was an error fetching de data from the CSV: {_OUTPUT_FILEPATH}")

if __name__ == "__main__":
    main()
