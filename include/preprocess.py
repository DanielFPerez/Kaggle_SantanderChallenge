"""
@Description:   Generic File for performing pre-processing steps

@Created:       03.04.2019

@author         Daniel F. Perez Ramirez

"""
# ===============================================================
# Import
# ===============================================================
import pandas as pd
from sklearn import preprocessing
import numpy as np


# ===============================================================
# Useful variables used by the module: Taken from Explore features
# ===============================================================
# Columns having 1 possible value
COL_1_VALUED = ['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41',
                'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41',
                'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46',
                'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3',
                'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1',
                'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
                'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3']
# Columns having 2 possible values
COL_2_VALUED = ['ind_var1_0', 'ind_var1', 'ind_var5_0', 'ind_var5', 'ind_var6_0', 'ind_var6', 'ind_var8_0', 'ind_var8',
                'ind_var12_0', 'ind_var12', 'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto', 'ind_var13_largo_0',
                'ind_var13_largo', 'ind_var13_medio_0', 'ind_var13_medio', 'ind_var13', 'ind_var14_0', 'ind_var14',
                'ind_var17_0', 'ind_var17', 'ind_var18_0', 'ind_var18', 'ind_var19', 'ind_var20_0', 'ind_var20',
                'ind_var24_0', 'ind_var24', 'ind_var25_cte', 'ind_var26_0', 'ind_var26_cte', 'ind_var26', 'ind_var25_0',
                'ind_var25', 'ind_var29_0', 'ind_var29', 'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31',
                'ind_var32_cte', 'ind_var32_0', 'ind_var32', 'ind_var33_0', 'ind_var33', 'ind_var34_0', 'ind_var34',
                'ind_var37_cte', 'ind_var37_0', 'ind_var37', 'ind_var39_0', 'ind_var40_0', 'ind_var40', 'ind_var41_0',
                'ind_var39', 'ind_var44_0', 'ind_var44', 'num_var6_0', 'num_var6', 'num_var8', 'num_var13_medio_0',
                'num_var13_medio', 'num_var18_0', 'num_var18', 'num_var20_0', 'num_var20', 'num_var29_0', 'num_var29',
                'num_var34_0', 'num_var34', 'num_var40', 'num_var39', 'num_var44', 'delta_imp_amort_var18_1y3',
                'delta_imp_amort_var34_1y3', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var33_1y3',
                'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_out_1y3', 'delta_num_reemb_var13_1y3',
                'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_out_1y3',
                'imp_reemb_var17_hace3', 'imp_reemb_var33_ult1', 'imp_trasp_var33_out_ult1', 'ind_var7_emit_ult1',
                'ind_var7_recib_ult1', 'ind_var10_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1',
                'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'num_var7_emit_ult1', 'num_meses_var13_medio_ult3',
                'num_reemb_var13_ult1', 'num_reemb_var17_hace3', 'num_reemb_var33_ult1', 'num_trasp_var17_in_ult1',
                'num_trasp_var17_out_ult1', 'num_trasp_var33_in_hace3', 'num_trasp_var33_out_ult1', 'saldo_medio_var29_hace3']
# Columns having 3 possible values
COL_3_VALUED = ['num_var1_0', 'num_var1', 'num_var8_0', 'num_var13_corto_0', 'num_var13_corto', 'num_var24', 'num_var33',
                'num_var40_0', 'num_var44_0', 'saldo_var6', 'saldo_var13_medio', 'saldo_var18', 'saldo_var29',
                'saldo_var34', 'delta_imp_reemb_var17_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var33_in_1y3',
                'delta_num_reemb_var17_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var33_in_1y3',
                'imp_amort_var18_ult1', 'imp_amort_var34_ult1', 'imp_trasp_var17_in_hace3', 'num_aport_var33_ult1',
                'num_meses_var29_ult3', 'num_trasp_var17_in_hace3', 'num_trasp_var33_in_ult1', 'num_venta_var44_hace3',
                'saldo_medio_var13_medio_hace2', 'saldo_medio_var13_medio_ult1', 'saldo_medio_var13_medio_ult3']

# Candidates for normalization
# Columns having between 100 and 1000 possible values
COL_1k_VALUED = ['var3', 'imp_ent_var16_ult1', 'imp_op_var40_comer_ult1', 'imp_op_var40_comer_ult3', 'imp_op_var40_ult1',
                 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3',
                 'saldo_var1', 'saldo_var13_corto', 'saldo_var13_largo', 'saldo_var13', 'saldo_var14', 'saldo_var17',
                 'saldo_var20', 'saldo_var31', 'saldo_var40', 'saldo_var44', 'imp_aport_var13_hace3', 'imp_aport_var13_ult1',
                 'imp_var7_recib_ult1', 'num_var45_ult3', 'saldo_medio_var8_hace3', 'saldo_medio_var13_corto_hace3',
                 'saldo_medio_var13_corto_ult1', 'saldo_medio_var13_largo_hace2', 'saldo_medio_var13_largo_hace3',
                 'saldo_medio_var13_largo_ult1', 'saldo_medio_var13_largo_ult3', 'saldo_medio_var17_ult1',
                 'saldo_medio_var17_ult3', 'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3']
# Columns having between 1001 and 5000 possible values
COL_5k_VALUED = ['saldo_var8', 'saldo_var12', 'saldo_var24', 'saldo_var26', 'saldo_var25', 'saldo_var37',
                 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'saldo_medio_var8_hace2', 'saldo_medio_var8_ult1',
                 'saldo_medio_var8_ult3', 'saldo_medio_var12_hace2', 'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1',
                 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_ult3']
# Columns having between 5001 and 10000 possible values
COL_10k_VALUED = ['imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1',
                  'imp_op_var41_comer_ult3', 'imp_op_var41_ult1', 'imp_op_var39_ult1', 'saldo_medio_var5_hace3']
# Columns having more than 10001 possible values
COL_upper10k_VALUED = ['saldo_var5', 'saldo_var30', 'saldo_var42', 'saldo_medio_var5_hace2', 'saldo_medio_var5_ult1',
                       'saldo_medio_var5_ult3', 'var38']

std_cols = ['ID', 'TARGET']
# ===============================================================
# Functions
# ===============================================================

def get_raw_data():
    """
    Loads the provided dataframe, removes the single valued columns and replaces a value in var3 according to the
    results in ../Explore_data
    :return: dataframe - (pandas dataframe)
    """
    # Load provided dataframe
    # Path is relative to the script it is run from
    raw_data_dir = './data/train.csv'

    dataframe = pd.read_csv(raw_data_dir)

    # Rremove the single valued columns
    dataframe = dataframe.drop(COL_1_VALUED, axis=1)

    # Remove unusual -99999 value from var3, either with zeros or with a positive value
    # Replacing the values with 0 wouldn't be good as there are only 75 zero values out of ~76000
    # We note that value 2 is the majority value, having 74k out of 76k hits
    dataframe[['var3']] = dataframe[['var3']].replace(-999999, 2)

    # Return dataframe
    return dataframe


def filter_datframe(dataframe, filter_case="None"):
    """
    Returns filtered dataframe
    :param dataframe: pandas dataframe to be filtered
    :param filter_case: string specifying the case to be filtered
    :return: filtered dataframe according to filter_case
    """

    if filter_case == "None":
        return dataframe

    elif filter_case == "low-valued":
        # Return 2-valued and 3-valued columns
        return dataframe[std_cols + COL_2_VALUED + COL_3_VALUED]

    elif filter_case == "low-valued-plus":
        # Return columns with 4 to 100 values
        feat_cols = [x for x in dataframe.columns if not x in std_cols]
        filter_out = COL_3_VALUED + COL_2_VALUED + COL_1k_VALUED + COL_5k_VALUED + COL_10k_VALUED + COL_upper10k_VALUED
        cols = [elem for elem in feat_cols if not elem in filter_out]
        return dataframe[std_cols + cols]

    elif filter_case == "middle-valued":
        # Return columns with 100 to 5000 values
        return dataframe[std_cols + COL_1k_VALUED + COL_5k_VALUED]

    elif filter_case == "middle-valued-plus":
        # Return columns with 100 to 10000
        return dataframe[std_cols + COL_1k_VALUED + COL_5k_VALUED + COL_10k_VALUED]

    elif filter_case == "high-valued":
        # Return columns with more than 5000
        return dataframe[std_cols + COL_10k_VALUED + COL_upper10k_VALUED]

    elif filter_case == "high-valued-plus":
        # Return columns with more than 1000 values
        return dataframe[std_cols + COL_5k_VALUED + COL_10k_VALUED + COL_upper10k_VALUED]

    print("DataFrame normalized")


def norm_transform_dataframe(dataframe, norm_case, scale_factor=1):
    """
    Perform normalization to dataframe
    :param dataframe: (pd.dataFrame) dataframe to be normalized
    :param norm_case: (string) "mean" or "minmax"
    :param scale_factor: (int) optional factor when using min max normalization
    :return:
    """
    feat_cols = [x for x in dataframe.columns if not x in std_cols]

    df_to_norm = dataframe[feat_cols]

    if norm_case == "mean":
        # Perform mean normalization
        normalized_df = (df_to_norm - df_to_norm.mean()) / df_to_norm.std()
        for elem in std_cols:
            normalized_df[elem] = dataframe[elem]
        return normalized_df
    elif norm_case == 'minmax':
        # Perform min max normalization and optionally use a scaling factor
        normalized_df = (df_to_norm-df_to_norm.min())/(df_to_norm.max()-df_to_norm.min())*scale_factor
        for elem in std_cols:
            normalized_df[elem] = dataframe[elem]
        return normalized_df

    print("DataFrame normalized")



