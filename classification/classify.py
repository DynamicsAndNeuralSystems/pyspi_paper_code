from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import sys
import os
from math import floor, ceil
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# from Oliver
import numpy as np
import pandas as pd
from copy import copy

# Classifier stuff
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, permutation_test_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
# mpl.use('TkAgg') # uncomment if you're having problems with multithreading

###############################################################################
# Helper functions

def classify_by_SPI(pipe, SPI_name, SPI_data_x, class_labels, sample_IDs, train_prop=0.9, test_prop=0.1, train_num=None, test_num=None, LOOCV=False, num_resamples=30, num_nulls=100):
    
    # Drop any features containing 100% NaN
    SPI_data_x[:,~np.isnan(SPI_data_x).any(axis=0)]
    
    # Convert any remaining NaN to numbers
    SPI_data_x = np.nan_to_num(SPI_data_x)
    
    if LOOCV:
        # Leave one out cross validation
        splitter = LeaveOneGroupOut()
    else:
        # Supply train_prop and test_prop unless train_num and test_num are defined to StratifiedShuffleSplit
        if train_num is None:
            # Stratified shuffle splitting
            splitter = StratifiedShuffleSplit(n_splits = num_resamples,
                                            train_size = train_prop,
                                            test_size = test_prop)
        else:
            # Stratified shuffle splitting
            splitter = StratifiedShuffleSplit(n_splits = num_resamples,
                                            train_size = train_num,
                                            test_size = test_num)

    resampled_test_scores = []
    resample_indices = []
    for i, (train_index, test_index) in enumerate(splitter.split(SPI_data_x, class_labels, groups=sample_IDs)):
        resample_indices.append(i+1)
        # Split data
        X_train, X_test = SPI_data_x[train_index], SPI_data_x[test_index]
        y_train, y_test = class_labels[train_index], class_labels[test_index]

        # Fit and score
        pipe.fit(X_train, y_train)
        accuracy_for_resample = accuracy_score(y_test, pipe.predict(X_test))
        resampled_test_scores.append(accuracy_for_resample)

    # Fit real and null data
    if LOOCV:
        real_avg_score, null_score, pvalue_real = permutation_test_score(pipe, 
                                                            SPI_data_x, 
                                                            class_labels, 
                                                            cv=splitter, 
                                                            n_jobs = 4,
                                                            groups = sample_IDs,
                                                            n_permutations=num_nulls, 
                                                            scoring="accuracy")
    else:
        real_avg_score, null_score, pvalue_real = permutation_test_score(pipe, 
                                                                    SPI_data_x, 
                                                                    class_labels, 
                                                                    cv=splitter, 
                                                                    n_jobs = 4,
                                                                    n_permutations=num_nulls, 
                                                                    scoring="accuracy")
    
    main_res_df = (pd.DataFrame(resampled_test_scores, columns=["Accuracy"])
                   .assign(Resample_Number = resample_indices)
                   .assign(SPI = SPI_name))
    
    # Create a real and null dataframe
    null_res_df = (pd.DataFrame(null_score.T, columns=["Null_Accuracy"])
                   .assign(SPI = SPI_name))
                
    return main_res_df, null_res_df

# Define function to run classification across all SPIs
def classify_across_all_SPIs(pipe, full_pyspi_data_x, class_labels, sample_IDs, train_prop=0.9, test_prop=0.1, train_num=None, test_num=None, LOOCV=False, num_resamples=30, num_nulls=100):
    
    # Drop any features containing 100% NaN
    full_pyspi_data_x[:,~np.isnan(full_pyspi_data_x).any(axis=0)]
    
    # Convert any remaining NaN to numbers
    full_pyspi_data_x = np.nan_to_num(full_pyspi_data_x)
    
    if LOOCV:
        # Leave one out cross validation
        splitter = LeaveOneGroupOut()
    else:
        # Supply train_prop and test_prop unless train_num and test_num are defined to StratifiedShuffleSplit
        if train_num is None:
            # Stratified shuffle splitting
            splitter = StratifiedShuffleSplit(n_splits = num_resamples,
                                            train_size = train_prop,
                                            test_size = test_prop)
        else:
            # Stratified shuffle splitting
            splitter = StratifiedShuffleSplit(n_splits = num_resamples,
                                            train_size = train_num,
                                            test_size = test_num)

    resampled_test_scores = []
    resample_indices = []
    for i, (train_index, test_index) in enumerate(splitter.split(full_pyspi_data_x, class_labels, groups=sample_IDs)):
        resample_indices.append(i+1)
        # Split data
        X_train, X_test = full_pyspi_data_x[train_index], full_pyspi_data_x[test_index]
        y_train, y_test = class_labels[train_index], class_labels[test_index]

        # Fit and score
        pipe.fit(X_train, y_train)
        accuracy_for_resample = accuracy_score(y_test, pipe.predict(X_test))
        resampled_test_scores.append(accuracy_for_resample)
    
    # Fit real and null data
    if LOOCV:
        real_score, null_score, pvalue_real = permutation_test_score(pipe, 
                                                                    full_pyspi_data_x, 
                                                                    class_labels, 
                                                                    cv=splitter, 
                                                                    n_jobs = 4,
                                                                    groups = sample_IDs,
                                                                    n_permutations=num_nulls, 
                                                                    scoring="accuracy")
    else:
        real_score, null_score, pvalue_real = permutation_test_score(pipe, 
                                                                    full_pyspi_data_x, 
                                                                    class_labels, 
                                                                    cv=splitter, 
                                                                    n_jobs = 4,
                                                                    n_permutations=num_nulls, 
                                                                    scoring="accuracy")
    
    # Create a real and null dataframe
    main_res_df = (pd.DataFrame(resampled_test_scores, columns=["Accuracy"])
                   .assign(Resample_Number = resample_indices))
    null_res_df = pd.DataFrame(null_score.T, columns=["Null_Accuracy"])
                
    return main_res_df, null_res_df

# Define function to iterate over SPIs and measure classification  accuracy
def process_dataset(pyspi_data, pipe, SPI_directionality, study_metadata, by_subject=False, train_prop=0.9, test_prop=0.1, train_num=None, test_num=None, LOOCV=False, num_resamples=30, num_nulls=1000):
    
    # Define SPIs
    SPIs = pyspi_data.SPI.unique().tolist()

    # Main accuracy
    main_SPI_wise_res_list = []
    null_SPI_wise_res_list = []
    filtered_SPI_list = []

    # Iterate over each SPI for 30 resamples
    for spi in SPIs:
        print(f"Now processing SPI: {spi}")
        this_SPI_directionality = SPI_directionality.query("SPI == @spi").Directionality.tolist()[0]
        
        this_SPI_filtered_list = []
        # Look for directed vs undirected SPIs
        for sample_ID in pyspi_data.Sample_ID.unique().tolist():
            
            SPI_data = (pyspi_data
                        .query("SPI == @spi & Sample_ID == @sample_ID")
                        .pivot(index=["Node_from", "group"],
                                columns="Node_to",
                                values="value"))

            # Check if matrix is symmetrical
            if this_SPI_directionality == "Undirected":
                
                # If synmetrical, take the upper half of the triangle and pivot back to long
                SPI_data_filtered = (SPI_data
                                    .where(np.triu(np.ones(SPI_data.shape), k=1)
                                            .astype(np.bool))
                                    .stack()
                                    .reset_index())
                SPI_data_filtered.columns = ['Node_from', 'group', 'Node_to','value']
                
                # Add SPI and sample info
                SPI_data_filtered["SPI"] = spi
                SPI_data_filtered["Sample_ID"] = sample_ID
                filtered_SPI_list.append(SPI_data_filtered)
                this_SPI_filtered_list.append(SPI_data_filtered)
            
            else: 
                # Otherwise, just drop self-connections and append to df_list
                SPI_data_filtered = (SPI_data
                                    .stack()
                                    .reset_index()
                                    .query("Node_from != Node_to"))
                SPI_data_filtered.columns = ['Node_from', 'group', 'Node_to','value']
                
                # Add SPI and sample info
                SPI_data_filtered["SPI"] = spi
                SPI_data_filtered["Sample_ID"] = sample_ID
                filtered_SPI_list.append(SPI_data_filtered)
                this_SPI_filtered_list.append(SPI_data_filtered)
                
        # Combine SPI data passing filtered
        
        if by_subject:
            this_SPI_filtered = (pd.concat(this_SPI_filtered_list, axis=0)
                                 .rename(columns = {"Sample_ID": "Unique_ID"})
                                 .merge(study_metadata, on="Unique_ID", how="left"))
            
            SPI_data_to_classify = (this_SPI_filtered
                                    .assign(Node_Pair = lambda x: x["Node_from"] + "_" + x["Node_to"])
                                    .drop(["Node_from", "Node_to"], axis=1)
                                    .pivot(index = ["Unique_ID", "Sample_ID", "group"],
                                        columns = "Node_Pair",
                                        values = "value"))
        else:
            this_SPI_filtered = (pd.concat(this_SPI_filtered_list, axis=0)
                                 .merge(study_metadata, on="Sample_ID", how="left"))
            
            SPI_data_to_classify = (this_SPI_filtered
                                    .assign(Node_Pair = lambda x: x["Node_from"] + "_" + x["Node_to"])
                                    .drop(["Node_from", "Node_to"], axis=1)
                                    .pivot(index = ["Sample_ID", "group"],
                                        columns = "Node_Pair",
                                        values = "value"))
        
        if len(SPI_data_to_classify.index) > 0:
            # Extract just the X data
            if by_subject:
                sample_IDs = SPI_data_to_classify.index.get_level_values(1).to_numpy()
                class_labels = SPI_data_to_classify.index.get_level_values(2).to_numpy()
            else:
                sample_IDs = SPI_data_to_classify.index.get_level_values(0).to_numpy()
                class_labels = SPI_data_to_classify.index.get_level_values(1).to_numpy()
            SPI_data_x = SPI_data_to_classify.reset_index(drop=True).to_numpy()
            
            # Call get_accuracy
            SPI_res_main, SPI_res_null = classify_by_SPI(pipe = pipe, 
                                                        SPI_name = spi,
                                                        SPI_data_x = SPI_data_x,
                                                        sample_IDs = sample_IDs,
                                                        class_labels = class_labels, 
                                                        LOOCV = LOOCV,
                                                        train_prop = train_prop,
                                                        test_prop = test_prop, 
                                                        train_num=train_num,
                                                        test_num=test_num,
                                                        num_resamples = num_resamples,
                                                        num_nulls=num_nulls)
            
            # Append results to lists
            main_SPI_wise_res_list.append(SPI_res_main)
            null_SPI_wise_res_list.append(SPI_res_null)
            
    main_SPI_wise_res = pd.concat(main_SPI_wise_res_list, axis=0).reset_index()
    null_SPI_wise_res = pd.concat(null_SPI_wise_res_list, axis=0).reset_index()

    ###########################################################################
    
    # Combine filtered SPI dataframes into a full dataframe
    if by_subject:
        full_pyspi_data = (pd.concat(filtered_SPI_list, axis=0)
                            .reset_index(drop=True)
                            .assign(Node_Pair = lambda x: x["Node_from"] + "_" + x["Node_to"])
                            .assign(SPI_Node_Pair = lambda x: x["Node_Pair"] + "_" + x["SPI"])
                            .drop(["Node_from", "Node_to", "SPI", "Node_Pair"], axis=1)
                            .rename(columns = {"Sample_ID": "Unique_ID"})
                            .merge(study_metadata, on="Unique_ID", how="left")
                            .drop_duplicates()
                                .pivot(index = ["Unique_ID", "Sample_ID", "group"],
                                    columns = "SPI_Node_Pair",
                                    values = "value"))

    else:
        full_pyspi_data = (pd.concat(filtered_SPI_list, axis=0)
                       .reset_index(drop=True)
                       .assign(Node_Pair = lambda x: x["Node_from"] + "_" + x["Node_to"])
                        .assign(SPI_Node_Pair = lambda x: x["Node_Pair"] + "_" + x["SPI"])
                        .drop(["Node_from", "Node_to", "SPI", "Node_Pair"], axis=1)
                        .drop_duplicates()
                        .pivot(index = ["Sample_ID", "group"],
                                columns = "SPI_Node_Pair",
                                values = "value"))

    if len(full_pyspi_data.index) > 0:
        print("Now running all SPIs together!")
        # Extract just the X data
        if by_subject:
            sample_IDs = full_pyspi_data.index.get_level_values(1).to_numpy()
            class_labels = full_pyspi_data.index.get_level_values(2).to_numpy()
        else:
            sample_IDs = full_pyspi_data.index.get_level_values(0).to_numpy()
            class_labels = full_pyspi_data.index.get_level_values(1).to_numpy()
        full_pyspi_data_x = full_pyspi_data.reset_index(drop=True).to_numpy()
        
        # Call get_accuracy
        main_full_res, null_full_res = classify_across_all_SPIs(pipe = pipe, 
                                                                full_pyspi_data_x = full_pyspi_data_x,
                                                                class_labels = class_labels, 
                                                                train_prop = train_prop,
                                                                test_prop = test_prop, 
                                                                train_num=train_num,
                                                                test_num=test_num,
                                                                sample_IDs=sample_IDs,
                                                                LOOCV = LOOCV,
                                                                num_resamples = num_resamples,
                                                                num_nulls=num_nulls)
    # Return everything
    return (main_SPI_wise_res, null_SPI_wise_res, main_full_res, null_full_res)

###############################################################################
# Load data
SPI_directionality = pd.read_csv("../data/SPI_directionality.csv") # Change to where you have saved the SPI directionality file

# Command-line arguments to parse
parser = argparse.ArgumentParser(description='Process inputs for pairwise data preparation.')
parser.add_argument('--dataset_ID', default="BasicMotions", dest='dataset_ID')
args = parser.parse_args()
dataset_ID = args.dataset_ID

data_path = f"/path/to/data/on/cluster/{dataset_ID}/" # Change this to the path to the data on the cluster

if dataset_ID == "Rest_vs_Film_fMRI": 
    pyspi_data = (pd.read_feather(f"{data_path}/processed_data/{dataset_ID}_pyspi_filtered_for_classification.feather")
                       .rename(columns = {"Sample_ID": "Subject_ID",
                                          "Unique_ID": "Sample_ID"})
                       )
    metadata = pd.read_feather(f"{data_path}/{dataset_ID}_metadata.feather")
else:
    pyspi_data = pd.read_feather(f"{data_path}/processed_data/{dataset_ID}_pyspi_filtered_for_classification.feather")
    metadata = pd.read_feather(f"{data_path}/{dataset_ID}_sample_metadata.feather")

classes = pyspi_data.group.unique().tolist()

###############################################################################
# Define universal parameters
num_resamples = 30
# Use linear SVM
pipe = Pipeline([('scaler', StandardScaler()), 
                 ('SVM', SVC(kernel="linear"))])
num_nulls = 100

###############################################################################
# Run classifier for given dataset and save results
if not os.path.isfile(f"{data_path}/processed_data/{dataset_ID}_main_SPI_wise_acc.feather"):
    print("Running SVMs for {dataset_ID}")
    if dataset_ID=="BasicMotions":
        main_SPI_wise_res, null_SPI_wise_res, main_full_res, null_full_res = process_dataset(pyspi_data = pyspi_data,
                                                                                    pipe=pipe,
                                                                                    SPI_directionality = SPI_directionality, 
                                                                                    study_metadata=metadata,
                                                                                    train_prop = 0.5,
                                                                                    test_prop = 0.5,
                                                                                    LOOCV = False,
                                                                                    by_subject=False,
                                                                                    num_resamples = num_resamples,
                                                                                    num_nulls=num_nulls)
    elif dataset_ID=="SelfRegulationSCP1":
        main_SPI_wise_res, null_SPI_wise_res, main_full_res, null_full_res = process_dataset(pyspi_data = pyspi_data,
                                                                                    pipe=pipe,
                                                                                    SPI_directionality = SPI_directionality, 
                                                                                    study_metadata=metadata,
                                                                                    train_num = 268,
                                                                                    test_num = 293,
                                                                                    LOOCV = False,
                                                                                    by_subject=False,
                                                                                    num_resamples = num_resamples,
                                                                                    num_nulls=num_nulls)
    elif dataset_ID=="Rest_vs_Film_fMRI":
        main_SPI_wise_res, null_SPI_wise_res, main_full_res, null_full_res = process_dataset(pyspi_data = pyspi_data,
                                                                                    pipe=pipe,
                                                                                    SPI_directionality = SPI_directionality, 
                                                                                    study_metadata=metadata,
                                                                                    LOOCV = True,
                                                                                    by_subject=True,
                                                                                    num_resamples = num_resamples,
                                                                                    num_nulls=num_nulls)
        
    main_SPI_wise_res.to_feather(f"{data_path}/processed_data/{dataset_ID}_main_SPI_wise_acc.feather")
    null_SPI_wise_res.to_feather(f"{data_path}/processed_data/{dataset_ID}_null_SPI_wise_acc.feather")
    main_full_res.to_feather(f"{data_path}/processed_data/{dataset_ID}_main_full_acc.feather")
    null_full_res.to_feather(f"{data_path}/processed_data/{dataset_ID}_null_full_acc.feather")
