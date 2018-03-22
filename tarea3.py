# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import numpy as np
import files.persistence as prst
import text.documents as docd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

DEFAULT_DATA_DIR = "data/"

prst.change_data_defaul_dir(DEFAULT_DATA_DIR)

reviews, puntuaciones = prst.load_stored_imbd()
rreview = reviews[0:90]
pptun = puntuaciones[0:90]
idf_vect = []
# ngram range, stop words, smooth idf, use idf, sublinear tf, binary, alpha

transform = TfidfVectorizer(input='content', encoding='utf-8',
    decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None,
    tokenizer=docd.tokenizer_and_stemmig, analyzer='word', stop_words=docd.get_stop_words_set(), token_pattern='(?u)\b\w\w+\b',
    ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None,
    binary=False, dtype=np.int32, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
#transform = TfidfVectorizer(input='content')
transform.fit(rreview)
X = transform.transform(rreview)

naive_bayes = MultinomialNB()
naive_bayes.fit(X, pptun)
print(naive_bayes.predict(transform.transform([reviews[91]])))
print("Ex: ", puntuaciones[91])

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(input='content', encoding='utf-8',
    decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None,
    tokenizer=docd.tokenizer_and_stemmig, analyzer='word', stop_words=docd.get_stop_words_set(), token_pattern='(?u)\b\w\w+\b',
    min_df=1, max_features=None, vocabulary=None,
    binary=False, dtype=np.int32, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)),
    ('clf', MultinomialNB())])

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__alpha': (1.0, 2.0, 0.5)
}

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=1, verbose=3, refit='AUC', scoring=scoring)
grid_search_tune.fit(rreview, pptun)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)
print(grid_search_tune.best_score_)

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

results = grid_search_tune.cv_results_

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, 20)
ax.set_ylim(0.73, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_tfidf__max_df'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()