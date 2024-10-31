from random import Random
from scipy import rand
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, ConfusionMatrixDisplay

def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")
    file = open("./FineTuningResults/GridSearchResults.txt", "w")
    
    y_train = y_train.values.ravel()
    
    model = RandomForestClassifier()
    
    gridsearch_grid = {'n_estimators': np.arange(90, 110, 10),
               'max_features': np.arange(5, 8),
               'max_depth': np.arange(15, 25),
               'min_samples_split': np.arange(5, 10),
               'min_samples_leaf': np.arange(4, 12),
               'bootstrap': [True]}
    
    model = GridSearchCV(estimator=model, param_grid=gridsearch_grid, cv=2, n_jobs=8, scoring="accuracy", verbose=10)
    model.fit(X_train.values, y_train)
    score = model.fit(X_train.values, y_train)
    print("Accuracy: %s" % score.best_score_)
    print("Best Params: %s" % score.best_params_)
    file.write("Best Params: %s" % score.best_params_)
    
    file.close()
    
    

    
    

if __name__ == '__main__':
    main()