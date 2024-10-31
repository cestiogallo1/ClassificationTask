# Script che esegue random forest con gli hyperparametri ottenuti e fare testing

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")
    X_test = pd.read_csv("./Dataset/XTest.csv")
    y_test = pd.read_csv("./Dataset/YTest.csv")
    
    y_train = y_train.values.ravel()
    
    # Dopo il fine tuning uso i parametri per fare testing e faccio la stampa degli score
    model=RandomForestClassifier(n_estimators=100, min_samples_split=7, min_samples_leaf=5, max_features=7, max_depth=19, bootstrap=True, verbose=10)
    model.fit(X_train, y_train)
    model_prediction = model.predict(X_test)

    p =  precision_score(y_test, model_prediction)
    a = accuracy_score(y_test, model_prediction)
    f1 = f1_score(y_test, model_prediction)
    r = recall_score(y_test, model_prediction)
	
    cm=confusion_matrix(y_test,model_prediction)
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='')
    plt.show()

    print("Accuracy: "+str(a))
    print("Precision: "+str(p))
    print("F1: "+str(f1))

    print(cm)
    
    
if __name__ == '__main__':
    main()