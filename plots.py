import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

from helper import *

cols = ['overall_loss_rate', 'overall_throughput', 'overall_delay']
names = ['Loss Rate', 'Throughput', 'Delay']

dfs = []
for filename in os.listdir('data/dataframes'):
    if filename.endswith('.pkl'):
        df = pickle.load(open('data/dataframes/' + filename, 'rb'))

        lost_sum = 0; delay_sum = 0; throughput = []; delay = []
        received_list = list(df['recvd']); delay_list = list(df['delay'])
        for i, t in enumerate(df['send_time']):
            if received_list[i] == 0: lost_sum += 1
            delay_sum += delay_list[i]
            throughput.append(((i + 1) - lost_sum) / t)
            delay.append(delay_sum / (i + 1))
        
        df['overall_throughput'] = throughput
        df['overall_delay'] = delay

        df = df.dropna(); df = df.reset_index(drop=True); ranges = []; i = 0
        for feature in cols:
            first = np.percentile(df[feature], 1)
            last = np.percentile(df[feature], 99)
            ranges.append((first, last))
        
        for feature in cols:
            df = df[(df[feature] >= ranges[i][0]) & (df[feature] <= ranges[i][1])]
            df = df.reset_index(drop=True); i += 1
        
        df = df.reset_index(drop=True)
        df.name = filename[:-4]; dfs.append(df)

for i, feature in enumerate(cols):
    fig, ax = plt.subplots(figsize=(12, 8))
    for df in dfs: 
        ax.plot(df['send_time'], df[feature], label=df.name)
        ax.set_title(f'{names[i]} over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(names[i])
        ax.legend(loc='upper right')
    fig.savefig('data/figures/' + feature + '.png')
    plt.close(fig)