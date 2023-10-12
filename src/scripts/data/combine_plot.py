#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets from .csv file downloaded from tensorboard
seed_1234 = pd.read_csv("sept16_1234.csv")
seed_1234.drop('Wall time', inplace = True, axis = 1) # Delete Wall Time column from dataframe
seed1234 = seed_1234['Value'].ewm(span = 2000, adjust = True).mean() # Smoothing Data by using Exponential Moving Average (ewm()) panda function
seed_1234.Value = seed1234 # Replace the original reward value with smoothed value

seed_900 = pd.read_csv("sept18_900.csv")
seed_900.drop('Wall time', inplace = True, axis = 1) # Delete Wall Time column from dataframe
seed900 = seed_900['Value'].ewm(span = 2000, adjust = True).mean() # Smoothing Data by using Exponential Moving Average (ewm()) panda function
seed_900.Value = seed900 # Replace the original reward value with smoothed value

seed_800 = pd.read_csv("sept19_800.csv")
seed_800.drop('Wall time', inplace = True, axis = 1) # Delete Wall Time column from dataframe
seed800 = seed_800['Value'].ewm(span = 2000, adjust = True).mean() # Smoothing Data by using Exponential Moving Average (ewm()) panda function
seed_800.Value = seed800 # Replace the original reward value with smoothed value

seed_700 = pd.read_csv("sept19_700.csv")
seed_700.drop('Wall time', inplace = True, axis = 1) # Delete Wall Time column from dataframe
seed700 = seed_700['Value'].ewm(span = 2000, adjust = True).mean() # Smoothing Data by using Exponential Moving Average (ewm()) panda function
seed_700.Value = seed700 # Replace the original reward value with smoothed value

seed_600 = pd.read_csv("sept19_600.csv")
seed_600.drop('Wall time', inplace = True, axis = 1) # Delete Wall Time column from dataframe
seed600 = seed_600['Value'].ewm(span = 2000, adjust = True).mean() # Smoothing Data by using Exponential Moving Average (ewm()) panda function
seed_600.Value = seed600 # Replace the original reward value with smoothed value

# Concatinate all epochs into one
seed = pd.concat([seed_1234, seed_900, seed_800, seed_700, seed_600])
seed.rename(columns = {'Step': 'Step', 'Value':'Reward Value'}, inplace = True) # Renaming y axis from Value to Reward Value

# print(seed)

# Save concatinated dataframe to .csv file
seed.to_csv('out.csv')

# Plot the reward function with error bands
sns.set(style='darkgrid')
sns.lineplot(x="Step", y="Reward Value", data=seed, errorbar='pi')
plt.tight_layout(pad=0.5)
plt.savefig('trial1.png', dpi = 300, bbox_inches='tight')