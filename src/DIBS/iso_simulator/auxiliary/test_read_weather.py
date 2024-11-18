# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:24:26 2023

@author: Kata
"""
import pandas as pd
a = pd.read_csv("weather_data/DEU_BE_Berlin-Schonefeld.AP.103850_TMYx.2004-2018.epw", skiprows=8, header=None).drop('datasource', axis=1)
print(a)