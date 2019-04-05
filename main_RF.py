"""
@Description:   File for testing purposes

@Created:       04.04.2019

@author         Daniel F. Perez Ramirez

"""
# ===============================================================
# IMPORT
# ===============================================================
import include.preprocess as preprocess
import include.printAssistant as printAssistant
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
plt.interactive(False)
# ===============================================================
# Fix/Random Variables for Hyper-parameter search
# ===============================================================

# Set cases extracting different features from the aavailable data
# See include/preprocess.py
df_filter_cases = ['low-valued-plus', 'middle-valued', 'middle-valued-plus', 'high-valued']

# Set normalization case implemented
norm_case_cases = ['mean', 'minmax']

# Set number of PCA components to try out
# Choose 3 random values between 3 dimensions and 20 dimensions
pca_dimension_cases = np.random.randint(low=3, high=18, size=4)
# Append value 15 as was the highest performing PCA dimension during manual exploration
if 15 not in pca_dimension_cases:
    pca_dimension_cases = np.append(pca_dimension_cases, 15)

# Parameters for the random forest
# Number of trees in the RF
n_estimators = np.random.randint(2, 15, size=4)
if 10 not in n_estimators:
    n_estimators = np.append(n_estimators, 10)
# Max depth of tree
max_depth = np.random.randint(5, 40, size=5)
if 30 not in n_estimators:
    max_depth = np.append(max_depth, 30)

rf_grid = [(i, j) for i in n_estimators.tolist() for j in max_depth.tolist()]

# ===============================================================
# Log Variables
# ===============================================================
# Dictionary for storing best performance tree in the test set
rf_winner = {"logger_index": None,
             "filter_case": None,
             "norm_case": None,
             "pca_dim": 0,
             "n_trees": 0,
             "tree_depth": 0,
             "auc_val": 0.0,
             "auc_test":0.0,
             "rf_object": None}
roc_auc_tracker = 0.0
roc_auc_val_tracker = 0.0

# Pandas dataFrame for logging
out_cols = ["filter_case", "norm_case", "pca_dim", "n_trees", "tree_depth", "cv_mean_auc", "test_auc", "image_path"]
pd_logger = pd.DataFrame(columns=out_cols)
logger_index = 0

# Directory for saving winner
out_path_winner = "./data/output/"

# Directory for saving logger and plots
out_path_log = "./data/output/log/"


# ===============================================================
# Build-Train-Test
# ===============================================================

# Load a pre-cleaned the dataset
df_raw = preprocess.get_raw_cleaned_data()

# Iterate over filter cases
for feature_filter in df_filter_cases:
    # Subtract a subset of the feature columns from filtered dataset
    df_filtered = preprocess.filter_dataframe(dataframe=df_raw, filter_case=feature_filter)

    for norm_case in norm_case_cases:
        df_normalized = preprocess.norm_transform_dataframe(dataframe=df_filtered, norm_case=norm_case,
                                                            scale_factor=1000)

        for pca_dim in pca_dimension_cases:
            # Apply PCA
            df_pca = preprocess.apply_pca(dataframe=df_normalized, dimensions=pca_dim)

            # Split data
            X_train, X_test, y_train, y_test = preprocess.split_datagram_test_train(df_pca,
                                                                                    test_size=0.1)

            # Perform oversampling
            X_smot, y_smot = preprocess.perform_smote(X_train, y_train)

            for (estim, depth) in rf_grid:
                cv = StratifiedKFold(n_splits=8)
                classifier = RandomForestClassifier(n_estimators=estim, max_depth=depth, min_samples_split=600)

                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)

                cv_tracker = 0
                for train, test in cv.split(X_smot, y_smot):
                    classifier.fit(X_smot[train], y_smot[train])
                    probabilities = classifier.predict_proba(X_smot[test])
                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(y_smot[test], probabilities[:,1])
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    # Add result of cv fold to the final auc plot
                    printAssistant.subplot_rocauc_cvfold(fpr, tpr, roc_auc, cv_tracker)
                    # Increment cv tracker
                    cv_tracker += 1

                mean_tpr = np.mean(tprs, axis=0)
                mean_auc = auc(mean_fpr, mean_tpr)

                # Report maximum achieve average on cross validation
                if mean_auc > roc_auc_val_tracker:
                    print("Maximum cv mean auc found: ", mean_auc, " at logger index ", logger_index)

                std_auc = np.std(aucs)
                std_tpr = np.std(tprs, axis=0)

                # Evaluate on test set
                probs_test = classifier.predict_proba(X_test)
                fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probs_test[:, 1])
                roc_auc_test = auc(fpr_test, tpr_test)

                # Log result
                printAssistant.plot_total_rocauc(mean_tpr, mean_auc, std_auc, std_tpr, mean_fpr)
                img_path = out_path_log + "cv_roc_plot_" + str(logger_index) + ".png"
                printAssistant.save_auc_roc_plot(img_path)
                pd_logger.loc[logger_index] = {"filter_case": feature_filter,
                                               "norm_case": norm_case,
                                               "pca_dim": pca_dim,
                                               "n_trees": estim,
                                               "tree_depth": depth,
                                               "cv_mean_auc": mean_auc,
                                               "test_auc": None,
                                               "image_path": img_path}

                # Log winner rf if test performance achieved
                if roc_auc_test > roc_auc_tracker:
                    roc_auc_tracker = roc_auc_test
                    rf_winner["logger_index"] = logger_index
                    rf_winner["filter_case"] = feature_filter
                    rf_winner["pca_dim"] = pca_dim
                    rf_winner["n_trees"] = estim
                    rf_winner["tree_depth"] = depth
                    rf_winner["auc_val"] = mean_auc
                    rf_winner["auc_test"] = roc_auc_test
                    rf_winner["rf_object"] = classifier
                    print("New best AUC over test set found: ", roc_auc_test)
                    printAssistant.save_auc_roc_plot(out_path_winner+"rf_winner.png")

                # Close plot to be able to plot after each iteration
                printAssistant.close_plot()

# Save log table
pd_logger.to_csv(out_path_log+"log.csv")

# Save winner model
pickle_out = out_path_winner+"rf_winner.pkl"
with open(pickle_out, 'wb') as file:
    pickle.dump(rf_winner, file, protocol=pickle.HIGHEST_PROTOCOL)

print("That's it, enjoy the results!")
