# Import libraries
import numpy as np
import pandas as pd
import random
import time

# import silence_tensorflow.auto
# import tensorflow as tf

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import pickle

# Pickling the trained model
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)
    return loaded_model

# Function for plotting graph of train score and cross validation score
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(6, 5))

    axes.set_title(title)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, _fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean-train_scores_std,
                      train_scores_mean+train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean-test_scores_std,
                      test_scores_mean+test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-',
              color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-',
              color="g", label="Cross-validation score")
    axes.legend(loc="best")
    plt.savefig(title + ".png")


# plot_learning_curve(classifier, "SVM", X_train, y_train,
#                     axes=None, ylim=(0.6, 1.01))

fig, axes = plt.subplots(1, 1, figsize=(10, 15))
# accuracy = tf.metrics.Accuracy()
epochs = 250

################### Support Vector Machine #####################

# Fitting SVM classifier
def classify_before_GA_SVM(X_train, X_test, y_train, y_test):
    classifier = SVC(C=10, gamma=0.01, kernel="rbf")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return accuracy_score(y_true=y_test, y_pred=y_pred)


def fitness_score_SVM(chromosome, X_train, X_test, y_train, y_test, save = False):
    classifier = SVC(C=10, gamma=0.01, kernel="rbf")
    temp = np.where(chromosome, True, False)
    classifier.fit(X_train.iloc[:, temp], y_train)
    y_pred = classifier.predict(X_test.iloc[:, temp])
    if save == True:
        save_model(classifier, "./finalized_svm.sav")
    return accuracy_score(y_true=y_test, y_pred=y_pred)

##################### LOGISTIC REGRESSION #####################

# LR Classifier
# def classify_before_GA_LR(X_train, X_test, y_train, y_test):
#     classifier = LogisticRegression(
#         penalty='l2', solver='lbfgs', max_iter=1500)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     return accuracy_score(y_true=y_test, y_pred=y_pred)


# def fitness_score_LR(chromosome, X_train, X_test, y_train, y_test, save = False):
#     classifier = LogisticRegression(
#         penalty='l2', solver='lbfgs', max_iter=1500)
#     temp = np.where(chromosome, True, False)
#     classifier.fit(X_train.iloc[:, temp], y_train)
#     y_pred = classifier.predict(X_test.iloc[:, temp])
#     if save == True:
#         save_model(classifier, "./finalized_lr.sav")
#     return accuracy_score(y_true=y_test, y_pred=y_pred)

##################### NEURAL NETWORK #####################

# def classify_before_GA_NN(X_train, X_test, y_train, y_test):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(units=64, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=32, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=16, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=8, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#     model.compile(optimizer='adam', loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     model.reset_states()
#     model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
#     y_pred = model.predict(X_test)

#     y_pred = (y_pred > 0.5)
#     accuracy.reset_states()
#     accuracy.update_state(y_pred, y_test)
#     acc = accuracy.result().numpy()
#     return acc


# def fitness_score_NN(chromosome, X_train, X_test, y_train, y_test):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(units=64, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=32, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=16, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=8, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#     model.compile(optimizer='adam', loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     temp = np.where(chromosome, True, False)
#     model.reset_states()
#     model.fit(X_train.iloc[:, temp], y_train,
#               batch_size=32, epochs=epochs, verbose=0)
#     y_pred = model.predict(X_test.iloc[:, temp])

#     y_pred = (y_pred > 0.5)
#     accuracy.reset_states()
#     accuracy.update_state(y_pred, y_test)
#     acc = accuracy.result().numpy()
#     return acc
