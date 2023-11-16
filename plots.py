import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from helper import *

dfs = [pickle.load(open('dataframes/' + filename, 'rb')) for filename in os.listdir('dataframes')]

for feature in dfs[0].columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for df in dfs: axes[0].plot(df.index, df[feature], label=df.name)
    axes[0].set_title(f'{feature} over Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(feature)
    axes[0].legend()

    for df in dataframes: sns.kdeplot(df[feature], ax=axes[1], label=df.name)
    axes[1].set_title(f'Frequency Distribution of {feature}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.show()