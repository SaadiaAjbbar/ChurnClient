import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#****************** EXPLORATION DE DATA **********************************
path_raw="data/raw/rawFile.csv"
df=pd.read_csv(path_raw)
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

#****************** MATPLOTLIB ET SEABORN ******************************
