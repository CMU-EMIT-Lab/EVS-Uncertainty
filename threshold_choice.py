import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
from ExtremeLy import extremely as ely
import matplotlib.pyplot as plt

import time
from pyextremes import plot_threshold_stability, plot_mean_residual_life


v1 = 395.84

ts = []
names = []

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

params = pd.read_csv('parameters.csv')
poredf = pd.read_csv('poredf.csv')

for i in range(len(params)):

    row = params.iloc[i]

    vol = row['Volume (mm^3)']

    pdf = poredf[poredf.Name == row.Name]

    pdf = pdf[['ESD', 'Label Index']]
    pdf['Label Index'] = pd.to_datetime(pd.date_range(start = '2000', end = '2001', periods = len(pdf))) #equally spaced in 1 year
  

    pdf = pdf.set_index('Label Index')

    pdf = pdf.squeeze()

    plot_mean_residual_life(pdf)
    plot_threshold_stability(pdf, return_period = v1/vol, r = 0, progress=True)

    plt.show(block=False)
    time.sleep(1)

    t = input('Threshold: ')
    if t == '':
        break
    else:
        params.iloc[i]['Threshold (um)'] = t
        params.iloc[i]['Shape'] = 0
        params.iloc[i]['Scale'] = 0
        params.iloc[i]['STDShape'] = 0
        params.iloc[i]['STDScale'] = 0
        params.iloc[i]['CovShapeScale'] = 0
        params.iloc[i]['Rate (1/mm^3)'] = 0
        params.iloc[i]['STDRate'] = 0
        params.to_csv('parameters_new.csv')
    