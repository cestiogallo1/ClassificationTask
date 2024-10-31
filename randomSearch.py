from random import Random
from xml.sax.handler import feature_external_ges
from scipy import rand
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, ConfusionMatrixDisplay

def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")
    file = open("./FineTuningResults/RandomSearchResultsTest.txt", "w")
    
    y_train = y_train.values.ravel()
    
    model = RandomForestClassifier()
    
    # Ho scelto i migliori parametri secondo diverse pagine web e documentazione      
    random_grid = {'n_estimators': np.arange(100, 1000, 100),
               'max_depth': np.arange(5, 50),
               'max_features': np.arange(1, 10),
               'min_samples_split': np.arange(2, 10),
               'min_samples_leaf': np.arange(2, 50),
               }

    model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 2, random_state=42, n_jobs = 8, scoring="accuracy", verbose=10)   
    
    # Testo e salvo in un file i risultati
    score = model.fit(X_train.values, y_train)
    print("Accuracy: %s" % score.best_score_)
    print("Best Params: %s" % score.best_params_)
    
    file.write("Best Params: %s" % score.best_params_)
    file.close()
    
if __name__ == '__main__':
    main()