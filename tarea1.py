# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import math, operator
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import text.documents as docd

def get_most_closer(new_text, kmean, vectorizer, vectores, number=3):
    vector_new = vectorizer.transform([new_text])
    cluster_new = kmean.predict(vector_new)[0]
    cercanos = get_closers(kmean.labels_, cluster_new)
    similitudes = get_similitudes(vectores, cercanos, vector_new)
    result = []
    for i in range(0, len(similitudes)):
        if not i < number:
            break
        result.append(similitudes[i])
    return result

def calcular_producto_escalar(elemento1, elemento2):
    suma = 0
    for i in range(0, elemento1.shape[1]):
        suma += elemento1[0,i] * elemento2[0,i]
    return suma

def square_root_element(elemento1):
    suma = 0
    for i in range(0, elemento1.shape[1]):
        suma += math.pow(elemento1[0,i], 2)
    return math.sqrt(suma)

def get_similitud(elemento1, elemento2):
    escalar = 0
    pww1 = 0
    pww2 = 0
    for i in range(0, elemento1.shape[1]):
        escalar += elemento1[0,i] * elemento2[0,i]
        pww1 += math.pow(elemento1[0,i], 2)
        pww2 += math.pow(elemento2[0,i], 2)
    return escalar / (pww1 * pww2)


def get_similitudes(vectors, index_list, element):
    result = []
    for index in index_list:
        sim = get_similitud(element, vectors[index])
        result.append((index, sim))
    return sorted(result, key=operator.itemgetter(1), reverse=True)

def get_closers(train_clases, cluster):
    result = []
    for i in range(0, len(train_clases)):
        if train_clases[i] == cluster:
            result.append(i)
    return result


categories =['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'comp.windows.x', 'sci.space']

newsgroups_train_data = fetch_20newsgroups(subset='train', categories=categories).data
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
print(len(newsgroups_train_data))

vectorizer = TfidfVectorizer(stop_words=docd.get_stop_words_set(), tokenizer=docd.tokenizer_and_stemmig)
vectorizer.fit(newsgroups_train_data)
X = vectorizer.transform(newsgroups_train_data)

print(X[0])

kmeans = KMeans(n_clusters=20, random_state=0).fit(X)

print(kmeans.cluster_centers_)

print(kmeans.predict(vectorizer.transform([newsgroups_test.data[0]])))

mejores = get_most_closer(newsgroups_test.data[0], kmeans, vectorizer, X, 3)

print(">---------------------------------------------<")
print("Nuevo: ")
print(newsgroups_test.data[0])
print(">---------------------------------------------<")
for i in range(0, len(mejores)):
    print("Puntuacion: ", mejores[i][1])
    print(newsgroups_train_data[mejores[i][0]])
    print(">---------------------------------------------<")


