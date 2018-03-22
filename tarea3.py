# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import numpy as np
import files.persistence as prst
import text.documents as docd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=1, verbose=3)
grid_search_tune.fit(rreview, pptun)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
