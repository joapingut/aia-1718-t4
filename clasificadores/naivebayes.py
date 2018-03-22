# -*- coding: utf-8 -*-
__author__ = 'Joaquin'

import math
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import text.documents as docd

class NaiveBayes():

    def __init__(self):
        self.log_pc = None
        self.log_pct = None
        self.k = None
        self.vectorizer = None
        self.clases = None
        self.entrenado = False

    def fit(self, textos, soluciones, k):
        self.k = k
        self.vectorizer, vectorizado = vectorizar(textos)
        V = vectorizado[0].shape[1]
        separados, self.clases = group_by_class(vectorizado, soluciones)
        modV = calcular_V(vectorizado)
        self.log_pc = prob_pc(separados, len(soluciones))
        self.log_pct = prob_ptc(separados, k, V, modV)
        self.entrenado = True

    def predict(self, titular):
        if self.entrenado:
            t_vec = self.vectorizer.transform([titular])[0]
            best = None
            c_best = None
            for c in range(0, len(self.clases)):
                suma = 0
                for i in range(0, t_vec.shape[1]):
                    suma += self.log_pct[c][i] * t_vec[0,i]
                candidate = self.log_pc[c] + suma
                if best is None or candidate > best:
                    best = candidate
                    c_best = self.clases[c]
            return c_best
        else:
            None

    def valida(self, validacion, resultado):
        suma = 0
        total = len(validacion)
        for i in range(0, total):
            predicho = self.predict(validacion[i])
            if predicho == resultado[i]:
                suma += 1
        return suma/total

def vectorizar(textos):
    vectorizer = CountVectorizer(stop_words=docd.get_stop_words_set_spanish())
    vectorizer.fit(textos)
    X = vectorizer.transform(textos)
    return (vectorizer, X)


def group_by_class(conjunto, soluciones):
    equivalente = []
    result = []
    for i in range(0, len(soluciones)):
        solucion = soluciones[i]
        index = None
        if solucion in equivalente:
            index = equivalente.index(solucion)
        else:
            index = len(equivalente)
            equivalente.append(solucion)

        if len(result) > index:
            result[index].append(conjunto[i])
        else:
            result.append([conjunto[i]])
    return (result, equivalente)

def calcular_V(vectores):
    return np.sum(vectores)

def prob_pc(elements, total):
    result = []
    for clase in elements:
        result.append(np.log10(len(clase)/total))
    return result

#Clase 0 -> ejemplos
#[[[000000], [000000]], [[11111], [11111]]]
def prob_ptc(elements, k, V, modV):
    result = []
    for clase in elements:
        l_tct = []
        tcs = 0
        for ejemplo in clase:
            for t in range(0, V):
                if len(l_tct) > t:
                    l_tct[t] = l_tct[t] + ejemplo[0,t]
                else:
                    l_tct.append(ejemplo[0,t])
                tcs += ejemplo[0,t]
        log_l = []
        for i in range(0, V):
            p_pro = (l_tct[i] + k) / (tcs + (k * modV))
            log_l.append(np.log10(p_pro))
        result.append(log_l)
    return result
