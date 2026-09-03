import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


path_clean = "data/processed/cleaned_data.csv"

df = pd.read_csv(path_clean)

print("****** Shape des donnees ******")
print(df.shape)

print("****** premiers lignes ******")
print(df.head())




X_cluster = df.drop(
    columns=["Churn", "customerID"],
    errors="ignore"
)

print("****** Variables pour le clustering ******")
print(X_cluster.columns)


# types de variables (numerical ou categorical)
numeric_features = X_cluster.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_cluster.select_dtypes(
    include=["object", "category"]
).columns

print("****** Variables numeriques ******")
print(numeric_features)

print("****** Variables catergoricals ******")
print(categorical_features)


# PREPROCESSING 

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

X_scaled = preprocessor.fit_transform(X_cluster)

print("****** Donnees apres preprocessing ******")
print(X_scaled.shape)


# ****************** ELBOW METHOD ******************

inertias = []

K_range = range(2, 11)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    inertias,
    marker="o"
)

plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertia")
plt.title("Méthode Elbow")

plt.grid(True)
plt.show()

# SILHOUETTE SCORE 
silhouette_scores = []

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)


plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score selon le nombre de clusters")

plt.grid(True)
plt.show()
