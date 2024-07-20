import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sympy import symbols, Eq, solve

obs_cir_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [6, 4, 0.8], [6.4, 7.2, 1.0], [4.8, 0.8, 0.4],
					     [2, 6, 0.3]])


list_samp = np.ravel(obs_cir_list)
print(type(list_samp))
df = []
dd = np.concatenate((list_samp, [3 ,5, 4]), axis=0)
print(dd.shape)


if df == []:
    df = np.concatenate(([list_samp], [list_samp]))
else:
    df = np.concatenate([df, [list_samp]])
print(df[:, :])

np.savetxt('samp.csv', obs_cir_list, delimiter=',')