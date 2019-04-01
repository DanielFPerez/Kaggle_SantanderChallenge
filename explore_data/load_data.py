"""
    Module description 
"""


import os
import pandas as pd


def get_directories():
    # Get Working Directory 
    src_dir = os.getcwd()
    print("Working directory: \t\t", src_dir)

    # Data directory
    data_dir = os.path.join(src_dir, "data")
    print("Path for data files: \t\t", data_dir)

    # Source path for training data
    train_data_src = os.path.join(data_dir, "train.csv")
    print("Path to trainign data file: \t", train_data_src)
    
    return src_dir, data_dir, train_data_src
    

def get_dataframe(path):
    pd_train = pd.read_csv(path)
    return pd_train