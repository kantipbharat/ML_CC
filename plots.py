import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from helper import *

cols = [['cwnd', 'ssthresh']]
cols += [['ewma_inter_send', 'ratio_inter_send']]
cols += [['ewma_inter_arr', 'ratio_inter_arr']]
cols += [['min_rtt', 'ratio_rtt']]
cols += [['throughput', 'delay']]
cols += [['loss_rate', 'overall_loss_rate']]

dfs = []
for filename in os.listdir('data/dataframes'):
    if filename.endswith('.pkl'):
        df = pickle.load(open('data/dataframes/' + filename, 'rb'))
        df = df.dropna(); df = df.reset_index(drop=True)
        ranges = []; i = 0

        for feature_set in cols:
            for feature in feature_set:
                first = np.percentile(df[feature], 1)
                last = np.percentile(df[feature], 99)
                ranges.append((first, last))
        
        for feature_set in cols:
            for feature in feature_set:
                df = df[(df[feature] >= ranges[i][0]) & (df[feature] <= ranges[i][1])]
                df = df.reset_index(drop=True); i += 1

        df = df.reset_index(drop=True)
        df.name = filename[:-4]; dfs.append(df)

for name, feature_set in enumerate(cols):
    for feature in feature_set:
        fig, ax = plt.subplots(nrows=1, ncols=len(feature_set), figsize=(12 * len(feature_set), 10))
        for i, feature in enumerate(feature_set):
            for df in dfs: 
                ax[i].plot(df['send_time'], df[feature], label=df.name)
            ax[i].set_title(f'{feature} over Time')
            ax[i].set_xlabel('Time')
            ax[i].set_ylabel(feature)
            ax[i].legend(loc='upper right')
        fig.savefig('data/figures/' + str(name + 1) + '.png')