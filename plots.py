import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from helper import *

cols = ['cwnd', 'cwnd_order']
cols += ['ewma_inter_send', 'min_inter_send']
cols += ['ewma_inter_arr', 'min_inter_arr']
cols += ['min_rtt']
cols += ['ssthresh', 'throughput', 'max_throughput', 'loss_rate', 'overall_loss_rate', 'delay']
cols += ['ratio_inter_send', 'ratio_inter_arr', 'ratio_rtt']

dfs = []
for filename in os.listdir('dataframes'):
    print(filename)
    if filename.endswith('.pkl'):
        df = pickle.load(open('dataframes/' + filename, 'rb'))
        df = df.dropna(); df = df.reset_index(drop=True)
        df.name = filename[:-4]; dfs.append(df)

for feature in COLUMNS:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for df in dfs: axes[0].plot(df.index, df[feature], label=feature)
    axes[0].set_title(f'{feature} over Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(feature)
    axes[0].legend()

    for df in dfs: sns.kdeplot(df[feature], ax=axes[1], label=df.name)
    axes[1].set_title(f'Frequency Distribution of {feature}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Density')
    axes[1].legend()
    plt.show()