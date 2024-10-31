# Script per comparazione modelli con i parametri di base

from numpy import mean
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")
    
    # Uso ravel per rendere l'array 1-D
    y_train = y_train.values.ravel()
   

    # dataset bilanciato, confronto con l'accuracy i modelli, funzione ritorna appunto l'accuracy
    results = cross_val_score(LogisticRegression(), X_train, y_train, cv=10, n_jobs=8)
    print("Regressione logistica -> accuracy: ", mean(results))
    
    results = cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=10, n_jobs=8)
    print("Decision tree -> accuracy: ", mean(results))
    
    results = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=10, n_jobs=8)
    print("Random Forest -> accuracy: ", mean(results))
    
    results = cross_val_score(AdaBoostClassifier(), X_train, y_train, cv=10, n_jobs=8)
    print("AdaBoost -> accuracy: ", mean(results))
    
    results = cross_val_score(GradientBoostingClassifier(), X_train, y_train, cv=10, n_jobs=8)
    print("GradientBoosting -> accuracy: ", mean(results))
    
    results = cross_val_score(XGBClassifier(), X_train, y_train, cv=10, n_jobs=8)
    print("XGB -> accuracy: ", mean(results))
    
if __name__ == '__main__':
    main()