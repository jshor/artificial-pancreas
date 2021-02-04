# -*- coding: utf-8 -*-
"""
@author: created by Sowmya Myneni and updated by Dijiang Huang
@author: modified for use by Josh Shor
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense

# Variable Setup
# Batch Size
BatchSize=10
# Epohe Size
NumEpoch=10

scenario = "A"

scenarios = {
    "A": {
        "train": ["1", "3"],
        "test": ["2", "4"]
    },
    "B": {
        "train": ["1", "2"],
        "test": ["1"]
    },
    "C": {
        "train": ["1", "2"],
        "test": ["1", "2", "3"]
    }
}


# builds the scenario from the given scenario key (A, B, C)
def build_scenario (scenario):
    print("Running Scenario {0}...".format(scenario))
    
    print(scenarios[scenario])
    
    train_subsets = scenarios[scenario]["train"]
    test_subsets = scenarios[scenario]["test"]
    
    train_file = "Training-a{0}.csv".format("-a".join(train_subsets))
    test_file = "Testing-a{0}.csv".format("-a".join(test_subsets))
    
    trainset = pd.read_csv(train_file, header=None)
    testset = pd.read_csv(test_file, header=None)
    
    # gather the unique values of all three subclass cols
    columns = [
        get_unique_col(trainset, testset, 1),
        get_unique_col(trainset, testset, 2),
        get_unique_col(trainset, testset, 3)    
    ]

    X_train, y_train = transform_dataset(trainset, columns)
    X_test, y_test = transform_dataset(testset, columns)
    
    # feature scaling
    sc = StandardScaler()
    
    X_train = sc.fit_transform(X_train)  # Scaling to the range [0,1]
    X_test = sc.fit_transform(X_test)

    return X_train, y_train, X_test, y_test


# transforms the dataset using OneHotEncoder
# this requires columns to be a list of lists containing unique values to map to
# (e.g., [['tcp', 'icmp', 'udp'], ['private', 'remote_job'], ['S0', 'REJ']])
def transform_dataset (dataset, columns):
    X = dataset.iloc[:, 0:-2].values
    label_column = dataset.iloc[:, -2].values
    y = []
    
    for i in range(len(label_column)):
        if label_column[i] == 'normal':
            y.append(0)
        else:
            y.append(1)
    
    # Convert ist to array
    y = np.array(y)
    
    # The following code work Python 3.7 or newer
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories=columns), [1,2,3])],    # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
        remainder='passthrough'                         # Leave the rest of the columns untouched
    )
    X = np.array(ct.fit_transform(X), dtype=np.float)

    return X, y


# returns a unique list of values from the column in the dataframe with the given index
def get_unique_col (a, b, index):
    a_uniq = list(set(a.iloc[:, index].values))
    b_uniq = list(set(b.iloc[:, index].values))
    
    return list(set(a_uniq + b_uniq))


# initializes a new keras classifier instance for training
def init_classifier (X_train, y_train):
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
    # rectified linear unit activation function (ReLU)
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
    
    # Adding the second hidden layer, 6 nodes
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer, 1 node, 
    # sigmoid on the output layer is to ensure the network output is between 0 and 1
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN, 
    # Gradient descent algorithm “adam“, Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    # This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“, Reference: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


def fit_model (classifier):
    # Fitting the ANN to the Training set
    # Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
    # add verbose=0 to turn off the progress report during the training
    # To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
    classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch)
    
    # evaluate the keras model for the provided model and dataset
    loss, accuracy = classifier.evaluate(X_train, y_train)
    
    # print the loss and the accuracy of the model on the dataset
    print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))
    
    return classifierHistory


# evaluates the model and returns a confusion matrix in the form:
# [[TN, FP]
#  [FN, TP]]
def get_confusion_matrix (classifier, X_test):
    print("Predicting results...")
    
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.9)   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
    
    return confusion_matrix(y_test, y_pred)


# plots the results using matplotlib
def plot_results (scenario, classifier_history, stat_feat):
    title = "Model {0}".format(stat_feat)
    file_name = "Scenario {0} {1}.png".format(scenario, stat_feat)
    
    print("Creating {0}...".format(file_name))
    
    plt.plot(classifier_history.history[stat_feat])
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(file_name)
    plt.show()
    
    print("Saved {0}.".format(file_name))
    
    
X_train, y_train, X_test, y_test = build_scenario(scenario)

classifier = init_classifier(X_train, y_train)

classifier_history = fit_model(classifier)

cm = get_confusion_matrix (classifier, X_test)

print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

plot_results(scenario, classifier_history, "accuracy")

plot_results(scenario, classifier_history, "loss")
