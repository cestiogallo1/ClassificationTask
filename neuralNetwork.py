# Script che permette di creare una piccola rete neurale con Keras, e permette anche di farci fine tuning
# con l'utilizzo di RandomizedSearchCV e GridSearchCV
# Ogni model.add crea un hidden layer

from pickletools import optimize
import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

X_train = pd.read_csv("./Dataset/XTrain.csv")
y_train = pd.read_csv("./Dataset/YTrain.csv") 
X_test = pd.read_csv("./Dataset/XTest.csv")
y_test = pd.read_csv("./Dataset/YTest.csv")

def randomSearch():
    model = models.Sequential()
    
    model.add(layers.Dense(5, activation = 'relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(3, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model = KerasClassifier(model=model)
    
    random_grid = {
        "epochs": np.arange(5, 25),
        "batch_size": np.arange(1, 100),
        "optimizer": ["SGD", "Adadelta", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "Adagrad"]
    }
    
    model = RandomizedSearchCV(estimator=model, param_distributions = random_grid, n_iter = 15, cv = 2, random_state=42, n_jobs = 8, scoring="accuracy", verbose=10)
    
    score = model.fit(X_train.values, y_train)
    print("Accuracy: %s" % score.best_score_)
    print("Best Params: %s" % score.best_params_)
    
def gridSearch():
    model = models.Sequential()
    
    model.add(layers.Dense(5, activation = 'relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(3, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model = KerasClassifier(model=model)
    
    random_grid = {
        "epochs": np.arange(5, 25),
        "batch_size": np.arange(1, 40),
        "optimizer": ["SGD", "Adadelta", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "Adagrad"]
    }
    
    model = GridSearchCV(estimator=model, param_distributions = random_grid, n_iter = 15, cv = 2, random_state=42, n_jobs = 8, scoring="accuracy", verbose=10)
    
    score = model.fit(X_train.values, y_train)
    print("Accuracy: %s" % score.best_score_)
    print("Best Params: %s" % score.best_params_)
    
def testModel():
    model = models.Sequential()
    
    model.add(layers.Dense(5, activation = 'relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(3, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    
    # Alcuni hyper parametri, training, e successivamente test
    model.compile(loss = 'binary_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])
    history = model.fit(X_train, y_train, epochs = 18, batch_size = 34, validation_split = 0.2, verbose = 1)
    test_loss, test_pred = model.evaluate(X_test, y_test)
    
    print(test_pred)
    print(test_loss)
    
    

def main():
    #randomSearch()
    #gridSearch()
    testModel()
    

    

if __name__ == '__main__':
    main()