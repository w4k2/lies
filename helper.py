import os  # to list files
import re  # to use regex
import numpy as np
from sklearn import (
    neural_network,
    neighbors,
    svm,
    gaussian_process,
    tree,
    ensemble,
    naive_bayes,
    discriminant_analysis,
    model_selection,
)
from sklearn import datasets


def datasets():
    directory = "datasets/"
    files = np.array(
        [
            (directory + x, x[:-4])
            for x in os.listdir(directory)
            if re.match("^([a-zA-Z0-9])+\.csv$", x)
        ]
    )
    return files


def results():
    directory = "results/"
    files = np.array(
        [
            (
                directory + x,
                x[:-4],
                x[:-4].split("_")[0],
                x[:-4].split("_")[1],
                x[:-4].split("_")[2],
            )
            for x in sorted(os.listdir(directory))
            if re.match("^([a-zA-Z0-9_])+\.csv$", x)
        ]
    )
    return files


def classifiers():
    return {
        "Nearest Neighbors": neighbors.KNeighborsClassifier(3),
        "RBF SVM": svm.SVC(gamma=2, C=1, probability=True),
        "Decision Tree": tree.DecisionTreeClassifier(max_depth=5),
        "Random Forest": ensemble.RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1
        ),
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Linear SVM": svm.SVC(kernel="linear", C=0.025, probability=True),
    }


def ks():
    return [10, 20]


def p_s():
    return [0.05]
