
import pandas as pd
import sys
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
import scipy
import re
import numpy as np
import pandas as pd
from copy import copy


###############################################################################
# Load data

# Load list of SPIs used in the pyspi paper
SPI_list = pd.read_csv("../data/SPI_list.csv").SPI.tolist()

# Load BasicMotions data
# Define data paths
BM_dataset_ID = "BasicMotions"
BM_data_path = "../data/BasicMotions/" # Change if you have your data stored elsewhere
BM_output_data_path = BM_data_path + "processed_data/"
BM_metadata = pd.read_feather(f"{BM_data_path}/BasicMotions_sample_metadata.feather")

BM_pyspi_data = (pd.read_feather(f"{BM_output_data_path}/BasicMotions_pyspi_filtered.feather")
                 .merge(BM_metadata, on='Sample_ID', how='left')
                 .query("SPI in @SPI_list")
                 .rename(columns={"activity": "group"}))

# Find only SPIs that have non-constant values to keep and are present in all samples
SPIs_to_keep_nonconstant = BM_pyspi_data.groupby("SPI", as_index=False).apply(lambda x: len(x.value.unique()) > 1).reset_index().rename(columns={None:"Non_constant"}).query("Non_constant == True").SPI.tolist()
SPIs_to_keep_in_all = BM_pyspi_data.groupby("SPI", as_index=False).apply(lambda x: len(x.Sample_ID.unique()) == len(BM_pyspi_data.Sample_ID.unique())).reset_index().rename(columns={None:"All_Sample_IDs"}).query("All_Sample_IDs == True").SPI.tolist()

# Take intersection and apply as a filter
SPIs_to_keep = list(set(SPIs_to_keep_nonconstant) & set(SPIs_to_keep_in_all))
BM_pyspi_data = BM_pyspi_data.query("SPI in @SPIs_to_keep")

# Print SPIs that were dropped
print("SPIs dropped from BasicMotions:")
print(set(SPI_list) - set(SPIs_to_keep))

# Save pyspi_data to feather 
BM_pyspi_data.reset_index().to_feather(f"{BM_output_data_path}/BasicMotions_pyspi_filtered_for_classification.feather")

# SelfRegulationSCP1 data
EEG_dataset_ID = "SelfRegulationSCP1"
EEG_data_path = "../data/SelfRegulationSCP1/" # Change if you have your data stored elsewhere
EEG_output_data_path = EEG_data_path + "processed_data/"
EEG_metadata = pd.read_feather(f"{EEG_data_path}/SelfRegulationSCP1_sample_metadata.feather")
EEG_pyspi_data = (pd.read_feather(f"{EEG_output_data_path}/SelfRegulationSCP1_pyspi_filtered.feather")
                  .merge(EEG_metadata, on="Sample_ID", how="left")
                 .query("SPI in @SPI_list")
                 .rename(columns={"cortical": "group"}))

# Find only SPIs that have non-constant values to keep and are present in all samples
SPIs_to_keep_nonconstant = EEG_pyspi_data.groupby("SPI", as_index=False).apply(lambda x: len(x.value.unique()) > 1).reset_index().rename(columns={None:"Non_constant"}).query("Non_constant == True").SPI.tolist()
SPIs_to_keep_in_all = EEG_pyspi_data.groupby("SPI", as_index=False).apply(lambda x: len(x.Sample_ID.unique()) == len(EEG_pyspi_data.Sample_ID.unique())).reset_index().rename(columns={None:"All_Sample_IDs"}).query("All_Sample_IDs == True").SPI.tolist()

# Take intersection and apply as a filter
SPIs_to_keep = list(set(SPIs_to_keep_nonconstant) & set(SPIs_to_keep_in_all))
EEG_pyspi_data = EEG_pyspi_data.query("SPI in @SPIs_to_keep")

# Print SPIs that were dropped
print("SPIs dropped from SelfRegulationSCP1:")
print(set(SPI_list) - set(SPIs_to_keep))

# Save pyspi_data to feather 
EEG_pyspi_data.reset_index().to_feather(f"{EEG_output_data_path}/SelfRegulationSCP1_pyspi_filtered_for_classification.feather")

# Rest vs Film fMRI data
fMRI_dataset_ID = "Rest_vs_Film_fMRI"
fMRI_data_path = "../data/Rest_vs_Film_fMRI/" # Change if you have your data stored elsewhere
fMRI_output_data_path = fMRI_data_path + "processed_data/"
fMRI_metadata = pd.read_feather(f"{fMRI_data_path}/Rest_vs_Film_fMRI_metadata.feather")
fMRI_pyspi_data = (pd.read_feather(f"{fMRI_output_data_path}/Rest_vs_Film_fMRI_pyspi_filtered.feather")
                      .rename(columns = {"Sample_ID": "Unique_ID"})
                      .merge(fMRI_metadata, on="Unique_ID", how="left")
                      .query("SPI in @SPI_list")
                      .rename(columns={"Scan_Type": "group"}))


# Find only SPIs that have non-constant values to keep and are present in all samples
SPIs_to_keep_nonconstant = fMRI_pyspi_data.groupby("SPI", as_index=False).apply(lambda x: len(x.value.unique()) > 1).reset_index().rename(columns={None:"Non_constant"}).query("Non_constant == True").SPI.tolist()
SPIs_to_keep_in_all = fMRI_pyspi_data.groupby("SPI", as_index=False).apply(lambda x: len(x.Sample_ID.unique()) == len(fMRI_pyspi_data.Sample_ID.unique())).reset_index().rename(columns={None:"All_Sample_IDs"}).query("All_Sample_IDs == True").SPI.tolist()

# Take intersection and apply as a filter
SPIs_to_keep = list(set(SPIs_to_keep_nonconstant) & set(SPIs_to_keep_in_all))
fMRI_pyspi_data = fMRI_pyspi_data.query("SPI in @SPIs_to_keep")

# Print SPIs that were dropped
print("SPIs dropped from Rest_vs_Film_fMRI:")
print(set(SPI_list) - set(SPIs_to_keep))

# Save pyspi_data to feather 
fMRI_pyspi_data.reset_index().to_feather(f"{fMRI_output_data_path}/Rest_vs_Film_fMRI_pyspi_filtered_for_classification.feather")