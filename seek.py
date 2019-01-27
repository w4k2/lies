#!/usr/bin/env python
import helper as h
import numpy as np
import pandas as pd
import csv, scipy, json
import warnings

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
    print("%i collisions found" % len(collisions))

    collisions = sorted(collisions, key=lambda l: l[0], reverse=True)
    with open("collisions/%s.csv" % clf, "w") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["n_db", "cv_method", "measure", "p", "r", "dbs"])
        for row in collisions:
            spamwriter.writerow(row)
            # print row
