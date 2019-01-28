#!/usr/bin/env python
import helper as h
import numpy as np
import pandas as pd
import csv, scipy, json
import warnings
from scipy.stats import wilcoxon, ttest_ind


def trow(dbname, mean_scores, std_scores, truths, cid):
    line = "\\emph{%s}" % dbname

    for j, clf_b in enumerate(clfs):
        line += " & "
        if j == cid:
            line += "\\color{red!75!black} "
        line += "\\small %s %.3f" % (
            "\\bfseries" if truths[j] == True else "",
            mean_scores[j],
        )

    line += "\\\\\n"
    return line


warnings.filterwarnings("ignore")

np.set_printoptions(precision=3, suppress=True)

# Parameters
repetitions = 10
datasets = h.datasets()
clfs = h.classifiers().keys()
results = h.results()
measures = ["t", "w"]
cv_methods = ["k10", "k20", "k2x5"]
p_s = h.p_s()

for i, clf in enumerate(clfs):
    collisions = []
    col_n = 0
    print("---\n%s [%i]" % (clf, i))
    for measure in measures:
        for p in p_s:
            for cv_method in cv_methods:
                for r in range(repetitions):
                    db_count = 0
                    dbs = []
                    for dataset in datasets:
                        dbname = dataset[1]
                        filename = "jsons/%s_r%i_%s_p%i.json" % (
                            dbname,
                            r,
                            cv_method,
                            int(p * 100),
                        )
                        data = json.load(open(filename))
                        scores = data["mean"]
                        advs = data["adv_%s" % measure]
                        score_leader = np.argmax(scores)
                        measure_leaders = np.argwhere(advs == np.max(advs))

                        # Warunek uznania
                        is_leader = i in measure_leaders and len(measure_leaders) < 3
                        if is_leader:
                            dbs.append(dbname)

                    if len(dbs) > 2:
                        record = [len(dbs), cv_method, measure, p, r, ":".join(dbs)]
                        collisions.append(record)

                        print("Collision found")
                        filename = "coltabs/c%i_%i.tex" % (i, col_n)
                        text_file = open(filename, "w")

                        col_n += 1
                        print(filename)

                        print(measure, p, cv_method, r, dbs)

                        for dbname in dbs:
                            # Gathering data from all repetitions
                            overtable = pd.read_csv(
                                "results/%s_r%i_%s.csv" % (dbname, r, cv_method)
                            ).values

                            # Calculate mean scores and std_s
                            mean_scores = np.mean(overtable, axis=0)
                            std_scores = np.std(overtable, axis=0)

                            # Establish leader
                            leader_id = np.argmax(mean_scores)
                            leader_sample = overtable[:, leader_id]

                            # Compare dependency
                            truths = []
                            for j, clf_b in enumerate(clfs):
                                if j == leader_id:
                                    truths.append(True)
                                    continue

                                if measure == "w":
                                    _, p_w = wilcoxon(leader_sample, overtable[:, j])
                                else:
                                    _, p_w = ttest_ind(leader_sample, overtable[:, j])
                                truths.append(p_w > p)

                            truths = np.array(truths)

                            text_file.write(
                                trow(dbname, mean_scores, std_scores, truths, i)
                            )
                        text_file.write(
                            "%% %i dbs, r=%i, p=%.2f, %s, %s"
                            % (len(dbs), r, p, cv_method, measure)
                        )
                        text_file.close()

                        # exit()
    print("%i collisions found" % len(collisions))

    collisions = sorted(collisions, key=lambda l: l[0], reverse=True)
    with open("collisions/%s.csv" % clf, "w") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["n_db", "cv_method", "measure", "p", "r", "dbs"])
        for row in collisions:
            spamwriter.writerow(row)
