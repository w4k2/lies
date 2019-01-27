# Tests

We test sixteen datasets from UCI ML (in the `datasets` directory). We use four approaches of division:

- 10-fold CV
- 20-fold CV
- 30-fold CV
- 5 times 2-fold CV

We run experiments for seven classifiers:

- kNN (k = 3),
- RBF SVM,
- Linear SVM,
- Naive Bayes,
- Decision Tree,
- Random Forest,
- MLP

In addition, for each set and each division method, we perform ten repetitions, in each case teaching and testing classifiers on a shared division. In total, it gives us 78400 results of the classification. The results are stored in the `results` directory.

Due to the presence of such binary as multi-class problems, the result is the *accuracy* measure, supplemented by a standard deviation.

For each combination of division, repetition and data set methods, tests are calculated for two methods:

- Wilcoxon test,
- Student's T-test.

We employ three thresholds for each method:

- p = .9
- p = .95
- p = .99

The analytical module sought to combine the methods of division, repetition, used test and threshold, which for each classifier yielded at least three sets of data in which the selected algorithm is the best or belongs to a group of up to three best statistically dependent algorithms (when each of them is statistically independent from the same number of others).

The `plots` directory contains bar graphs for all cases detected in this way.
