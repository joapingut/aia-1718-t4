# -*- coding: utf-8 -*-

__author__ = 'Joaquin'


import files.persistence as prst

DEFAULT_DATA_DIR = "data/"

prst.change_data_defaul_dir(DEFAULT_DATA_DIR)

ret = prst.read_titulares_csv("titulares.csv", True)
print(ret[1])
