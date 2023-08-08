# Load modules
import numpy as np
import pandas as pd
import os
import pyarrow.feather as feather


def merge_calcs_into_df(dataset_ID,
                        proc_data_path,
                        pkl_data_path,
                        proc_lookup_table,
                        pairwise_feature_set = "pyspi",
                        pkl_file = "calc.pkl"):
    '''
    Merge all of the individual calc.pkl files from one dataset into one dataframe
    and write to a feather file.

            Parameters:
                    dataset_ID (str): Name of the given dataset to process, e.g. BasicMotions
                    proc_data_path (str): Path to the processed data folder where output should go
                    pkl_data_path (str): Path to folder where the individual pickle files are stored
                    proc_lookup_table (str): Dataframe that links the process name to what it measures in the MTS
                    pairwise_feature_set (str): Name of the pairwise feature set, e.g. pyspi
                    pkl_file (str): Name of the output from pyspi-distribute, e.g. calc.pkl

            Returns:
                    None
    '''
    
    # Check if feather data file already exists
    if not os.path.isfile(f"{proc_data_path}/{dataset_ID}_{pairwise_feature_set}.feather"):
        # Where individual pickle data files will be read in from
        input_np_data_path = pkl_data_path
        
        # Find subjects
        samples = os.listdir(input_np_data_path)
        
        # Initialise list for each subject's pyspi data
        samples_with_pyspi_data = []
        sample_data_list = []
        for sample in [sample for sample in samples if ".npy" not in sample]:
            try:
                # Load in subject's pyspi data and filter
                with open(input_np_data_path + sample + "/" + pkl_file, "rb") as f:
                    sample_data = dill.load(f)
                    
                # Widen to a matrix for each SPI, only keeping upper half when symmetrical
                df_list = []
                for SPI in sample_data.SPI.unique():
                    SPI_data = (sample_data
                                .query("SPI == @SPI")
                                .pivot(index="brain_region_from",
                                                                      columns="brain_region_to",
                                                                      values="value"))

                        
                    # Pivot and append to df_list
                    SPI_data_full = (SPI_data
                                        .stack()
                                        .reset_index()
                                        .query("brain_region_from != brain_region_to"))
                    SPI_data_full.columns = ['proc_from','proc_to','value']
                    
                    # Add SPI and sample info
                    SPI_data_full["SPI"] = SPI
                    SPI_data_full["Sample_ID"] = sample
                    
                    # Append to df_list
                    df_list.append(SPI_data_full)
                        
                        
                sample_data = pd.concat(df_list, axis=0)
                
                # Add Sample_ID column
                sample_data["Sample_ID"] = sample
                
                # Append to list of pyspi data
                samples_with_pyspi_data.append(sample)
                sample_data_list.append(sample_data)
            except Exception as e: 
                print("Error for " + sample)
                print(e)

        # Switch process IDs (e.g. proc-0) for names of what those represent in the given MTS (e.g. Channel 1)
        full_pyspi_res = pd.concat(sample_data_list).reset_index()
        full_pyspi_res_lookup = ((pd.merge(full_pyspi_res, 
                                      proc_lookup_table,
                                      how = "left",
                                      left_on = "proc_from",
                                      right_on = "proc")
                              .drop(["index", "proc_from", "proc"], axis=1)
                              .rename(columns={"Node": "Node_from"})
                              ).merge(proc_lookup_table,
                                      how="left",
                                      left_on = "proc_to",
                                      right_on = "proc")
                                      .drop(["proc_to", "proc"], axis=1)
                                      .rename(columns={"Node": "Node_to"})
                                      )

        # Write the full merged pyspi data to a feather file for the given dataset
        feather.write_feather(full_pyspi_res_lookup, f"{proc_data_path}/{dataset_ID}_{pairwise_feature_set}.feather", version=1)

##########################################################################################################
# BasicMotions
BM_data_path = "../data/BasicMotions/" # Change if you have your data stored elsewhere
BM_dataset_ID = "BasicMotions"
BM_proc_data_path = BM_data_path + "processed_data/"
BM_pkl_data_path = BM_data_path + "numpy_files/"

BM_proc_lookup_table = pd.DataFrame({"proc": ["proc-0", "proc-1", "proc-2", "proc-3", "proc-4", "proc-5"],
                                    "Node": ["Accelerometer_x", "Accelerometer_y", "Accelerometer_z", "Gyroscope_x", "Gyroscope_y", "Gyroscope_z"]})

merge_calcs_into_df(proc_data_path = BM_proc_data_path,
                    pkl_data_path = BM_pkl_data_path,
                    pkl_file = pkl_file,
                    pairwise_feature_set = pairwise_feature_set,
                    proc_lookup_table = BM_proc_lookup_table,
                    dataset_ID = BM_dataset_ID)

##########################################################################################################
# SelfRegulationSCP1 EEG
EEG_data_path = "../data/SelfRegulationSCP1/" # Change if you have your data stored elsewhere
EEG_dataset_ID = "SelfRegulationSCP1"
EEG_proc_data_path = EEG_data_path + "processed_data/"
EEG_pkl_data_path = EEG_data_path + "numpy_files/"

EEG_proc_lookup_table = pd.DataFrame({"proc": ["proc-0", "proc-1", "proc-2", "proc-3", "proc-4", "proc-5"],
                                    "Node": ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6"]})

merge_calcs_into_df(proc_data_path = EEG_proc_data_path,
                    pkl_data_path = EEG_pkl_data_path,
                    pkl_file = pkl_file,
                    pairwise_feature_set = pairwise_feature_set,
                    proc_lookup_table = EEG_proc_lookup_table,
                    dataset_ID = EEG_dataset_ID)

##########################################################################################################
# Rest vs Film fMRI
fMRI_data_path = "../data/Rest_vs_Film_fMRI/" # Change if you have your data stored elsewhere
fMRI_dataset_ID = "Rest_vs_Film_fMRI"
fMRI_proc_data_path = fMRI_data_path + "processed_data/"
fMRI_pkl_data_path = fMRI_data_path + "numpy_files/"

fMRI_proc_lookup_table = pd.DataFrame({"proc": ["proc-0", "proc-1", "proc-2", "proc-3", "proc-4", "proc-5", "proc-6"],
                                        "Node": ["Control", "Default", "DorsalAttention", "Limbic", "Somatomotor", "VentralAttention", "Visual"]})

merge_calcs_into_df(proc_data_path = fMRI_proc_data_path,
                    pkl_data_path = fMRI_pkl_data_path,
                    pkl_file = pkl_file,
                    pairwise_feature_set = pairwise_feature_set,
                    proc_lookup_table = fMRI_proc_lookup_table,
                    dataset_ID = fMRI_dataset_ID)

