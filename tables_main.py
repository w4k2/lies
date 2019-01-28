"""
Prepare global tables.
"""

import helper as h
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_ind
np.set_printoptions(suppress=True)

# Parameters
repetitions = 10
datasets = h.datasets()
clfs = h.classifiers().keys()
p_s = h.p_s()

cv_methods = {
    "k20": 20,
    "k10": 10,
    "k2x5": 10
}

# Iterate division methods
for cv_method in cv_methods:
    text_file = open("table_%s.tex" % cv_method, "w")

    # Number of samples per method
    samples = cv_methods[cv_method]

    # Iterating datasets
    for dbpath, dbname in datasets:

        # Gathering data from all repetitions
        overtable = [pd.read_csv("results/%s_r%i_%s.csv" % (
            dbname, repetition, cv_method
        )).values for repetition in range(repetitions)]
        overtable = np.array(overtable).reshape(repetitions * samples, 6)

        # Calculate mean scores and std_s
        mean_scores = np.mean(overtable, axis=0)
        std_scores = np.std(overtable, axis=0)

        # Establish leader
        leader_id = np.argmax(mean_scores)
        leader_sample = overtable[:, leader_id]

        # Compare dependency
        truths = []
        for j, clf_b in enumerate(clfs):
            if j==leader_id:
                truths.append([True for p in p_s])
                continue

            _, p_w = wilcoxon(leader_sample,
                              overtable[:, j])
            truths.append([p_w > p for p in p_s])

        truths = np.array(truths)
        #print(mean_scores, std_scores, truths)

        def row(dbname, mean_scores, std_scores, truths):
            line = "\\emph{%s}" % dbname

            for j, clf_b in enumerate(clfs):
                line += " &\n"
                line += "\\begin{tikzpicture}[baseline=-10pt]\n"
                #print(clf_b, truths[j])

                line += "\\node at (-.2,0)[circle,fill,inner sep=.5pt, %s]{};\n" % ("white" if truths[j,0] == False else "black")

                line += "\\node at (0,0)[circle,fill,inner sep=.5pt,%s, label=below:{\\small %s %.3f}]{};\n" % ("white" if truths[j,1] == False else "black", "\\bfseries" if np.sum(truths[j]) > 0 else "", mean_scores[j])

                line += "\\node at (.2,0)[circle,fill,inner sep=.5pt, %s]{};\n" % ("white" if truths[j,2] == False else "black")

                line += "\\end{tikzpicture}"

            line += "\\\\"
            return line

        a = row(dbname, mean_scores, std_scores, truths)
        text_file.write(a)
