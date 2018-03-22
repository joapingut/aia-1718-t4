# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS_ENGLISH = None
STOP_WORDS_SPANISH = None
STEMMMER = None

def get_stop_words_set():
    global STOP_WORDS_ENGLISH
    if STOP_WORDS_ENGLISH is None:
        STOP_WORDS_ENGLISH = set(stopwords.words('english'))
    return STOP_WORDS_ENGLISH

def get_stop_words_set_spanish():
    global STOP_WORDS_SPANISH
    if STOP_WORDS_SPANISH is None:
        STOP_WORDS_SPANISH = set(stopwords.words('spanish'))
    return STOP_WORDS_SPANISH

def get_stemmer():
    global STEMMMER
    if STEMMMER is None:
        STEMMMER = PorterStemmer()
    return STEMMMER

def tokenizer_and_stemmig(text):
    words = word_tokenize(text)
    wordsFiltered = []
    stemmer = get_stemmer()
    for w in words:
        wordsFiltered.append(stemmer.stem(w))
    return wordsFiltered