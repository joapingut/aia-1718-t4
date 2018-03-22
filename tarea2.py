# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import math, operator
import numpy as np
import clasificadores.naivebayes as nb
import files.persistence as per


DEFAULT_DATA_DIR = "data/"

per.change_data_defaul_dir(DEFAULT_DATA_DIR)

entrenamiento = per.read_titulares_csv("titulares.csv", False)
validacion = per.read_titulares_csv("titulares_val.csv", False)
print(entrenamiento[0])
print(entrenamiento[1])

print(nb.group_by_class(entrenamiento[0], entrenamiento[1]))

naiveB = nb.NaiveBayes()
naiveB.fit(entrenamiento[0], entrenamiento[1], 1)
print("Hello")
print(naiveB.predict("Detienen al ladrón de casas en la provincia de Asturias"))
print(naiveB.predict("Dimite el presidente de la delegación de Asturias"))
print(naiveB.predict("M. Rajoy es un mentiroso"))
print(naiveB.predict("¿Quíén es M. Rajoy?"))

print(naiveB.valida(validacion[0], validacion[1]))
