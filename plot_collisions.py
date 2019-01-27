#!/usr/bin/env python
import helper as h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, scipy, json
import warnings

warnings.filterwarnings("ignore")

np.set_printoptions(precision=3, suppress=True)

clfs = h.classifiers().keys()

tests = {"w": "Wilcoxon"}
for cid, clf in enumerate(clfs):
    print(clf)
    colors = ["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3", "#1E88E5"]
    colors.insert(cid, "#F44336")

    df = pd.read_csv("collisions/%s.csv" % clf)
    for row in df.values:
        print(row)
        _, cv_method, measure, p, r, dbs = row
        dbs = dbs.split(":")

        fig, ax = plt.subplots(figsize=(2 * _, 5))
        ax.set_title(
            "%s as a good solution \nRepetition %i, with %s cv using %s test with p=%.2f"
            % (clf, r, cv_method, tests[measure], 1 - p)
        )
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(_))
        ax.set_xticklabels(dbs)

        for i, dbname in enumerate(dbs):
            filename = "jsons/%s_r%i_%s_p%i.json" % (dbname, r, cv_method, int(p * 100))
            # print filename
            data = json.load(open(filename))
            # print data["mean"]

            for j, clf_score in enumerate(data["mean"]):
                ax.bar(i + 0.12 * j - 0.36, clf_score, width=0.1, color=colors[j])
                ax.errorbar(
                    i + 0.12 * j - 0.36,
                    clf_score,
                    data["std"][j],
                    linestyle="None",
                    color="black",
                    alpha=0.5,
                    linewidth=0.5,
                )

        plotfilename = "plots/%s_r%i_%s_%s_p%i.png" % (
            clf.replace(" ", "_").lower(),
            r,
            cv_method,
            measure,
            int(p * 100),
        )
        plt.savefig(plotfilename)
        plt.close("all")
