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


def plot_rco_auc(fpr, tpr, roc_auc):
    # Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.0])
    plt.ylim([-0.1, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# ===============================================================
#   Model Evaluation plot
# ===============================================================
def subplot_rocauc_cvfold(fpr, tpr, roc_auc, i):
    """
    Adds a subplot to the final ROC plot
    :param fpr: (ndarray) (1,n) false positive rate
    :param tpr: (ndarray) (1,n) true positive rate
    :param roc_auc: (sklearn.metrics) metric evaluator
    :param i:   (int) current cross validation fold
    :return:
    """
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


def plot_total_rocauc(mean_tpr, mean_auc, std_auc, std_tpr, mean_fpr):
    """
    Plots the final ROC after cross validation
    :param mean_tpr: mean of the true positive rates
    :param mean_auc: mean of the area under the curve from the ROC
    :param std_auc: standard deviation from AUC
    :param std_tpr: standard deviation of the TPR
    :param mean_fpr: mean of the false positive rates
    :return:
    """
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    mean_tpr[-1] = 1.0

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC')
    plt.legend(loc="lower right")


def save_auc_roc_plot(out_file):
    plt.savefig(out_file, format="png")

def close_plot():
    plt.close()