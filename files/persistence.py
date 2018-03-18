# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import pickle, os
import numpy as np

DEFAULT_DATA_DIRECTORY = "../data/"

def change_data_defaul_dir(ndir):
    global DEFAULT_DATA_DIRECTORY
    DEFAULT_DATA_DIRECTORY = ndir

def get_data_default_dir():
    return DEFAULT_DATA_DIRECTORY

def store_data(name, obj):
    f = open(DEFAULT_DATA_DIRECTORY + name + ".pkl","wb")
    pickle.dump(obj,f)
    f.close()

def load_data(name):
    f = open(DEFAULT_DATA_DIRECTORY + name + ".pkl","rb")
    obj = pickle.load(f)
    f.close()
    return obj

def read_imdb_csv(name, test = False):
    f = open(DEFAULT_DATA_DIRECTORY + name,"r", encoding='utf-8')
    line = f.readline() # ignore first line
    line = f.readline()
    reviews = []
    punts = np.empty((0,), dtype=np.uint8)
    while line:
        i = line.rindex(',')
        review = line[:i]
        punt = int(line[i+1:])
        if test:
            print("Review read: ", review)
            print("Puntuationread: ", punt)
        reviews.append(review)
        punts = np.append(punts, punt)
        line = f.readline()
    return (reviews, punts)

def read_titulares_csv(name, test = False):
    f = open(DEFAULT_DATA_DIRECTORY + name,"r", encoding='utf-8')
    line = f.readline() # ignore first line
    line = f.readline()
    reviews = []
    punts = np.empty((0,), dtype=np.uint8)
    clases = {'sociedad': 0, 'deporte':1, 'politica':2}
    while line:
        i = line.rindex(',')
        review = line[:i]
        punt = line[i+1:].strip()
        if test:
            print("Review read: ", review)
            print("categoria: ", punt)
        reviews.append(review)
        punts = np.append(punts, clases[punt])
        line = f.readline()
    return (reviews, punts)

def save_imdb_data(tuple_imdb):
    store_data("reviews", tuple_imdb[0])
    store_data("puntuaciones", tuple_imdb[1])

def load_stored_imbd():
    rreviews = load_data("reviews")
    rpunts = load_data("puntuaciones")
    return (rreviews, rpunts)