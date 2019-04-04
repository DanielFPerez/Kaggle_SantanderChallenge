"""
@Description:   Module for performing printing operations

@Created:       02.04.2019

@author         Daniel F. Perez Ramirez

"""
# ===============================================================
# Import
# ===============================================================
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ===============================================================
# Explore data plot
# ===============================================================
def get_seaborn_random_pairplot(datagram, cols, n_rand_elem):
    l_rand_elem = np.random.randint(0, len(cols), size=n_rand_elem)
    sublist = [cols[i] for i in l_rand_elem]
    print(sublist)
    pd_Sublist = datagram[sublist + ["TARGET"]]
    sns.pairplot(data=pd_Sublist, hue='TARGET')

def get_xcross_plot(datagram, cols=None, figsize=(20, 16), cmap='coolwarm'):
    if cols:
        pd_subdata = datagram[cols]
    else:
        pd_subdata = datagram

    subdata_corr = pd_subdata.corr()

    subdata_corr = subdata_corr.round(decimals=2)

    f, ax = plt.subplots(figsize=figsize)

    sns.heatmap(subdata_corr, linewidths=0.5, cmap=cmap, annot=True, ax=ax)




# ===============================================================
#
# ===============================================================
