# Script per il preprocessing sul dataset

from audioop import minmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Mutual information fra tra feature e label
def mutualInformation(X, y):
    bestfeatures = SelectKBest(score_func=mutual_info_classif, k=2)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print("MUTUAL INFORMATION SCORES")
    print(featureScores)
 
# Scaling con MinMaxNormalization    
def scaler(X_train, X_test):
    scaler = MinMaxScaler()
    xd = scaler.fit_transform(X_train)
    xd1 = scaler.transform(X_test)
    X_train = pd.DataFrame(xd, columns=X_train.columns)
    X_test = pd.DataFrame(xd1, columns=X_test.columns)
    return X_train, X_test
    

def main():
    df = pd.read_csv("./Dataset/Airlines.csv")

    # drop feature id non utile
    df = df.drop(columns=['id'])


    # encoding feature categoriche
    o = OrdinalEncoder()
    for index in df.columns:
        if df[index].dtype == 'O':
            df[index] = o.fit_transform(df[index].values.reshape(-1, 1))

    oldShape = df.shape[0]

    # rimozione outliers con l'utilizzo dello z-score
    # - computazione dello z-score per ogni colonna
    # - valore assoluto dello z-score
    # - (axis=1) si assicura che ogni riga per tutte le colonne soddisfi la condizione
    df = df[(np.abs(zscore(df)) < 3).all(axis=1)]
    print("Rimossi ", oldShape - df.shape[0], " outliers")

    # controllo presenza valori nulli
    print("INFORMAZIONI SUI DATI")
    print(df.info())

    # stampa informazioni sul dataset (compresi media e deviazione standard)
    print("\nINFORMAZIONI SULLE FEATURE")
    print(df.describe())

    # splitto il dataset
    y = df.pop('Delay')
    X = df

    # controllo dipendenza con Mutual Information
    print("Attendere Mutual Information...")
    #mutualInformation(X, y)
    df = df.drop(columns=['Length'])
    df = df.drop(columns=['DayOfWeek'])
    X = df
    print("Rimossi Length e DayOfWeek")

    print("Creazione e salvataggio training e test set (con balancing)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

    # oversampling con SMOTE
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    X_train, X_test = scaler(X_train, X_test)

    X_train.to_csv("./Dataset/XTrain.csv", index=False)
    y_train.to_csv("./Dataset/YTrain.csv", index=False)
    X_test.to_csv("./Dataset/XTest.csv", index=False)
    y_test.to_csv("./Dataset/YTest.csv", index=False)

    df = pd.read_csv("./Dataset/YTrain.csv")
    df.groupby('Delay').size().plot(kind='pie', autopct='%.2f', label='Delays')
    plt.savefig("./Immagini/PieChartDelayAfterBalancing.png")

if __name__ == '__main__':
    main()