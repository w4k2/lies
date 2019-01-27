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
p_s = h.p_s()

# Iterating results
for p in p_s:
    print(p)
    for result in results:
        # Get params for results
        dbname = result[2]
        repetition = int(result[3][1:])
        cv_method = result[4]
        filename = result[0]
        df = pd.read_csv(filename)

        # Showoff
        #print "Repetition %i with %s on %s (p=%.2f)" % (repetition, cv_method, dbname,1-p)

        # Calculate mean scores and std_s
        mean_scores = df.mean(axis=0).values
        std_scores = df.std(axis=0).values

        # Classifier classification
        adv_w = [0] * len(clfs)
        adv_t = [0] * len(clfs)

        # Going through classifiers
        for i, clf_a in enumerate(clfs):
            clf_acc = mean_scores[i]
            #print "\n---\n%s - ACC = %.3f" % (clf_a, clf_acc)
            leading = np.delete(mean_scores < clf_acc, i)
            #print leading

            acc_vector_leader = df[clf_a].values

            p_ws, p_ts = [], []
            for j, clf_b in enumerate(clfs):
                if j == i:
                    continue
                acc_vector_b = df[clf_b].values

                _, p_w = scipy.stats.wilcoxon(acc_vector_leader,acc_vector_b)
                _, p_t = scipy.stats.ttest_ind(acc_vector_leader,acc_vector_b)
                p_ws.append(p_w)
                p_ts.append(p_t)
            p_ws = np.array(p_ws)
            p_ts = np.array(p_ts)

            mask_p_ws = p_ws < p
            mask_p_ts = p_ts < p
            actual_p_ws = mask_p_ws * leading
            actual_p_ts = mask_p_ts * leading
            a_ws = sum(actual_p_ws)
            a_ts = sum(actual_p_ts)
            """
            print "| Wilcoxon:"
            print p_ws
            print mask_p_ws
            print actual_p_ws
            print a_ws
            print "\n| T-student"
            print p_ts
            print mask_p_ts
            print actual_p_ts
            print a_ts
            """

            adv_w[i] = int(a_ws)
            adv_t[i] = int(a_ts)

        # Collect and write analysis
        data = {
            "parameters": {
                "db": dbname,
                "repetition": repetition,
                "cv_method": cv_method,
            },
            "clfs": list(clfs),
            "mean": list(mean_scores),
            "std": list(std_scores),
            "adv_w": adv_w,
            "adv_t": adv_t
        }
        jsonname = "jsons/%s_r%i_%s_p%i.json" % (dbname, repetition, cv_method, int(p*100))
        #print jsonname
        with open(jsonname, 'w') as outfile:
            json.dump(data, outfile, indent=2, sort_keys=True)
