import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#****************** EXPLORATION DE DATA **********************************
path_raw="data/raw/rawFile.csv"
df=pd.read_csv(path_raw)
print("****** LES COLUMNS SONT:******")
print(df.columns)
print("***les premieres lignes sont:***")
print(df.head())

print("***les dernieres lignes sont:***")
print(df.tail())

print("****les infos genereaux comme nb columns ... sont:*****")
print(df.info())

print("*******les lignes et columns sont:**********")
print(df.shape)

print("*******la structure statistique est:**********")
print(df.describe())
print("*******la structure statistique des variables cagtegorials est:**********")
print(df.describe(include="object"))
#****************** NETTOYAGE DES DONNEES ******************************
print("******les duplicateds****")
print(df.duplicated())
print("******total des duplicateds****")
print(df.duplicated().sum())
print("******les manquants****")
print(df.isna())
print("******total des manquants****")
print(df.isna().sum())

#supprimer les valeurs manquants

df = df.drop_duplicates()
print(df.duplicated().sum())

print(df.dtypes)
# changer type de totalchange a float
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
print("*****le type de totalcharges apres correction***")
print(df["TotalCharges"].dtype)
#detecter les valeurs manquant dans totalcharges
print("********* LE NOMBRE DES VALEURS MANQUANTS DANS TOTALCHARGES****")
print(df["TotalCharges"].isnull().sum())
# if df["TotalCharges"].isna()==True:
#     print(df["TotalCharges"])
#remplacer les manquants dans totalcharges 
df["TotalCharges"]=df["TotalCharges"].fillna(
    df["TotalCharges"].mean()
)
print("********* LE NOMBRE DES VALEURS MANQUANTS DANS TOTALCHARGES apres remplissage****")

print(df["TotalCharges"].isna().sum())
print(df["TotalCharges"])
#verifier les categories des valeurs
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_columns:
    print(df[col].value_counts())

#analyser churn (pourcentage de chaque valeur de churn)
print("******Analyse de Churn ****")
print(df["Churn"].value_counts(normalize=True)*100)

#les columns numerics
colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
print(colonnes_numeriques)

# #****************** MATPLOTLIB ET SEABORN (distribution des columns numeriques)******************************
#distribution de churn
sns.countplot(data=df, x="Churn")
plt.title("distribution de churn")
plt.show()
for col in colonnes_numeriques:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()

# #*********************correlation**************************
corr = df.select_dtypes(include="number").corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matrice de correlation")
plt.show()
#***********Outliers************
sns.boxplot(x=df["MonthlyCharges"])
plt.title("Outliers - MonthlyCharges")
plt.show()

# # detecter outliers
for col in colonnes_numeriques:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()
#avec IQR
Q1 = df["TotalCharges"].quantile(0.25)
Q3 = df["TotalCharges"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["TotalCharges"] < lower_bound) |
    (df["TotalCharges"] > upper_bound)
]

print("Nombre d'outliers :", len(outliers))

