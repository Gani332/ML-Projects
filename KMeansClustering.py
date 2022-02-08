import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)  # .data is all of our features - scaling all our features down to 1 or -1 as they all have very large values - saves time
y = digits.target  # Getting our labels

k = len(np.unique(y))  # This can be k = 10 (10 digits) but if there's different values here's how you get it
features, samples = data.shape  # Getting instances and the amount of numbers we have that we are going to classify as well as the features


def bench_k_means(estimator, name,
                  data):  # We can train a bunch of different classifiers and score them by calling this function
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# This whole def function is a math model which helps the machine learning get the best training accuracy

clf = KMeans(n_clusters=k, init="random", n_init=10,)  # For more: https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
bench_k_means(clf, "1", data)

""" * homogeneity - checks if each cluster contains only members of a single class. ranges from 0-1 with 1 being the best score

* completeness - checks if all members of a given class are assigned to the same cluster (ranges from 0-1 again)

* v_measure - is the harmonic mean of homo. and compl.  (ranges from 0-1 again)

* adjusted_rand_score - similarity of the actual values and their predicitions, ignoring permutations and with chance normalization (range from -1 to 1 with -1 being bad, 1 being perfect and 0 being random)

* adjusted_mutual_info - agreement of the actual values and predictions, ignoring permutations (from 0-1 with 0 being random agreement and 1 being perfect agreement)

* silhouette_score -  This uses two values, the mean distance between a sample and all other points in the same class as well as the mean distance between a sample and all other points in the nearest cluster to calculate a score that ranges from -1 to 1, with the former being incorrect, and the latter standing for highly dense clustering. 0 indicates overlapping clusters.

"""