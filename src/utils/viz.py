from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_test, y_pred):
    conf_mat = pd.crosstab(y_test, y_pred)
    conf_mat = conf_mat.div(conf_mat.sum(axis=0))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, square=True, ax=ax)
    return fig, ax


def get_top_grams(corpus, n, gram=1):
    vec = CountVectorizer(ngram_range=(gram, gram), stop_words='english').fit(corpus)
    counts = vec.transform(corpus).sum(axis=0)
    freqs = [(word, counts[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
    sample_freqs = freqs[:n]
    return np.array(sample_freqs)