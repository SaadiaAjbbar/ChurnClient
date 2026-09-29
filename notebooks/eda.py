import pandas as pd
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

#****************** MATPLOTLIB ET SEABORN (distribution)******************************
sns.histplot(df["MonthlyCharges"], kde=True)
plt.title("Distribution des charges mensuelles")
plt.show()

sns.histplot(df["tenure"], kde=True)
plt.title("Distribution de tenure")
plt.show()
#*********************correlation**************************
corr = df.select_dtypes(include="number").corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matrice de correlation")
plt.show()
