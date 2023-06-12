# Load libraries

library(tidyverse)
library(glue)
library(reticulate)
library(icesTAF)

# Note: change this to an absolute path to use pyspi-distribute
data_path <- "../data/"

# Make sure you've cloned pyspi-distribute from github.com/anniegbryant/pyspi-distribute to access create_yaml_for_samples.R
pyspi_distribute_path <- "/path/to/pyspi_distribute/repo/"
# Create a YAML file for these .npy files
yaml_script_file <- glue("{pyspi_distribute_path}/create_yaml_for_samples.R")

################################################################################
# Helper functions
write_data_to_numpy <- function(sample_df) {
  
  sample_ID <- unique(sample_df$Sample_ID)
  # Convert to matrix
  sample_mat <- sample_df %>%
    dplyr::select(-Sample_ID, -Node) %>%
    as.matrix()
  
  # Save to numpy file
  np$save(glue("{numpy_file_path}/{sample_ID}"),
          np$array(sample_mat))
}

################################################################################
# BasicMotions smartwatch activity dataset

# Download data to your repository if you don't already have it
if (!file.exists(glue("{data_path}/BasicMotions/processed_data/BasicMotions_TS.feather"))) {
  # Download zipped file that contains ARFF files + extract zipped contents
  BasicMotions_zip <- glue("{data_path}/BasicMotions.zip")
  download.file("http://www.timeseriesclassification.com/Downloads/BasicMotions.zip", BasicMotions_zip, mode="wb")
  unzip(BasicMotions_zip, exdir=glue("{data_path}/BasicMotions/arff_files/"))
  
  # Lookup table for EEG channels and their corresponding positions
  dimensions_lookup <- data.frame(Dimension = paste0("Dimension", 1:6),
                                  Node = c("Accelerometer_x",
                                               "Accelerometer_y",
                                               "Accelerometer_z",
                                               "Gyroscope_x",
                                               "Gyroscope_y",
                                               "Gyroscope_z"))
  
  # Initialise a list of dataframes for the six dimensions
  dim_df_list <- list()
  
  # Iterate over each dimension and parse the time-series data
  for (dim in dimensions_lookup$Dimension) {
    dim_train <- foreign::read.arff(glue("{data_path}/BasicMotions/arff_files/BasicMotions{dim}_TRAIN.arff")) %>%
      dplyr::mutate(ID = row_number(),
                    Dimension = dim) %>%
      left_join(., dimensions_lookup) %>%
      dplyr::mutate(Split = "Train") %>%
      select(-Dimension) %>%
      tidyr::pivot_longer(cols=c(-activity, -ID, -Node, -Split),
                          values_to = "signal",
                          names_to = "timepoint") 
    
    dim_test <- foreign::read.arff(glue("{data_path}/BasicMotions/arff_files/BasicMotions{dim}_TEST.arff")) %>%
      dplyr::mutate(ID = row_number(),
                    Dimension = dim) %>%
      left_join(., dimensions_lookup) %>%
      dplyr::mutate(Split = "Test") %>%
      select(-Dimension) %>%
      tidyr::pivot_longer(cols=c(-activity, -ID, -Node, -Split),
                          values_to = "signal",
                          names_to = "timepoint") 
    
    dim_df_list[[dim]] <- plyr::rbind.fill(dim_train, dim_test)
  }
  
  # Concatenate the dimensions into one dataframe
  BasicMotions_TS <- do.call(plyr::rbind.fill, dim_df_list) %>%
    mutate(Sample_ID = paste0(Split,  "_", ID)) %>%
    group_by(activity, Sample_ID, ID, Node, Split) %>%
    # z-score each time series
    mutate(timepoint_num = row_number(),
           signal_z = scale(signal)) 
  
  # Save concatenated dataframe to a feather file
  TAF::mkdir(glue("{data_path}/BasicMotions/processed_data/"))
  feather::write_feather(BasicMotions_TS, glue("{data_path}/BasicMotions/processed_data/BasicMotions_TS.feather"))
  
} else {
  # Load the time-series dataframe if the feather file already exists
  BasicMotions_TS <- feather::read_feather(glue("{data_path}/BasicMotions/processed_data/BasicMotions_TS.feather"))
}

# Split by group
BasicMotions_split_sample  <- BasicMotions_TS %>%
  # Reshape from long to wide such that columns are timepoints
  pivot_wider(id_cols = c("Sample_ID", "Node"),
              names_from = timepoint_num, values_from = signal_z) %>%
  group_by(Sample_ID) %>%
  # Split into a list of dataframes per sample ID
  group_split()

# Write files to numpy arrays for pyspi processing
np <- import("numpy")
numpy_file_path <- as.character(glue("{data_path}/BasicMotions/numpy_files"))
TAF::mkdir(numpy_file_path)

# Write numpy files for each sample
1:length(BasicMotions_split_sample) %>%
  purrr::map(~ write_data_to_numpy(sample_df = BasicMotions_split_sample[.x][[1]]))

ID_var <- "Sample_ID"
dim_order <- "ps"
yaml_file_base <- "sample.yaml"

cmd <- glue("Rscript {yaml_script_file} --data_dir {as.character(numpy_file_path)} --numpy_file_base .npy --ID_var {ID_var} --dim_order {dim_order} --yaml_file {yaml_file_base}")
system(cmd)

# Create a metadata file
sample_metadata <- BasicMotions_TS %>%
  distinct(ID, Split, activity) %>%
  mutate(Sample_ID = paste0(Split, "_", ID), .keep="unused")
feather::write_feather(sample_metadata, glue("{data_path}/BasicMotions/BasicMotions_sample_metadata.feather"))


################################################################################
# SelfRegulationSCP1 EEG data

# Download data to your repository if you don't already have it
if (!file.exists(glue("{data_path}/SelfRegulationSCP1/processed_data/SelfRegulationSCP1_TS.feather"))) {
  # Download zipped file that contains ARFF files
  SelfRegulationSCP1_zip <- glue("{data_path}/SelfRegulationSCP1.zip")
  download.file("http://www.timeseriesclassification.com/Downloads/SelfRegulationSCP1.zip", SelfRegulationSCP1_zip, mode="wb")
  unzip(SelfRegulationSCP1_zip, exdir=glue("{data_path}/SelfRegulationSCP1/arff_files/"))
  
  # Lookup table for EEG channels and their corresponding positions
  dimensions_lookup <- data.frame(Dimension = paste0("Dimension", 1:6),
                                  Node = paste0("Channel_", 1:6))
  
  # Initialise a list of dataframes for the six dimensions
  dim_df_list <- list()
  
  # Iterate over each dimension and parse the time-series data
  for (dim in dimensions_lookup$Dimension) {
    dim_train <- foreign::read.arff(glue("{data_path}/SelfRegulationSCP1/arff_files/SelfRegulationSCP1{dim}_TRAIN.arff")) %>%
      dplyr::mutate(ID = row_number(),
                    Dimension = dim) %>%
      left_join(., dimensions_lookup) %>%
      dplyr::mutate(Split = "Train") %>%
      select(-Dimension) %>%
      tidyr::pivot_longer(cols=c(-cortical, -ID, -Node, -Split),
                          values_to = "signal",
                          names_to = "timepoint") 
    
    dim_test <- foreign::read.arff(glue("{data_path}/SelfRegulationSCP1/arff_files/SelfRegulationSCP1{dim}_TEST.arff")) %>%
      dplyr::mutate(ID = row_number(),
                    Dimension = dim) %>%
      left_join(., dimensions_lookup) %>%
      dplyr::mutate(Split = "Test") %>%
      select(-Dimension) %>%
      tidyr::pivot_longer(cols=c(-cortical, -ID, -Node, -Split),
                          values_to = "signal",
                          names_to = "timepoint") 
    
    dim_df_list[[dim]] <- plyr::rbind.fill(dim_train, dim_test)
  }
  
  # Concatenate the dimensions into one dataframe
  SelfRegulationSCP1_TS <- do.call(plyr::rbind.fill, dim_df_list) %>%
    mutate(Sample_ID = paste0(Split,  "_", ID)) %>%
    group_by(cortical, Sample_ID, ID, Node, Split) %>%
    # z-score each time series
    mutate(timepoint_num = row_number(),
           signal_z = scale(signal)) 
  
  # Save concatenated dataframe to a feather file
  TAF::mkdir(glue("{data_path}/SelfRegulationSCP1/processed_data/"))
  feather::write_feather(SelfRegulationSCP1_TS, glue("{data_path}/SelfRegulationSCP1/processed_data/SelfRegulationSCP1_TS.feather"))
  
} else {
  # Load the time-series dataframe if the feather file already exists
  SelfRegulationSCP1_TS <- feather::read_feather(glue("{data_path}/SelfRegulationSCP1/processed_data/SelfRegulationSCP1_TS.feather"))
}

# Split by group
SelfRegulationSCP1_split_sample  <- SelfRegulationSCP1_TS %>%
  # Reshape from long to wide such that columns are timepoints
  pivot_wider(id_cols = c("Sample_ID", "Node"),
              names_from = timepoint_num, values_from = signal_z) %>%
  group_by(Sample_ID) %>%
  # Split into a list of dataframes per sample ID
  group_split()

# Write files to numpy arrays for pyspi processing
np <- import("numpy")
numpy_file_path <- as.character(glue("{data_path}/SelfRegulationSCP1/numpy_files"))
TAF::mkdir(numpy_file_path)

1:length(SelfRegulationSCP1_split_sample) %>%
  purrr::map(~ write_data_to_numpy(sample_df = SelfRegulationSCP1_split_sample[.x][[1]]))

# Create a YAML file for these .npy files
ID_var <- "Sample_ID"
dim_order <- "ps"
yaml_file_base <- "sample.yaml"

cmd <- glue("Rscript {yaml_script_file} --data_dir {as.character(numpy_file_path)} --numpy_file_base .npy --ID_var {ID_var} --dim_order {dim_order} --yaml_file {yaml_file_base}")
system(cmd)

# Create a metadata file
sample_metadata <- SelfRegulationSCP1_TS %>%
  distinct(ID, Split, cortical) %>%
  mutate(Sample_ID = paste0(Split, "_", ID), .keep="unused")
feather::write_feather(sample_metadata, glue("{data_path}/SelfRegulationSCP1/SelfRegulationSCP1_sample_metadata.feather"))

################################################################################
# Rest vs film fMRI data
Yeo_parc_info <- read.table("../data/Yeo_parc_info.csv", header=T, sep=",")

# Download data to your repository if you don't already have it
if (!file.exists(glue("{data_path}/Rest_vs_Film_fMRI/processed_data/Rest_vs_Film_fMRI_TS_Yeo7.feather"))) {
  # Note: the file to download is 254.9MB; you can increase the timeout length accordingly if you need more time to download
  options(timeout = max(300, getOption("timeout")))
  Rest_vs_Film_fMRI_zip <- glue("{data_path}/Rest_vs_Film_fMRI.zip")
  download.file("https://figshare.com/ndownloader/articles/12971162/versions/2", Rest_vs_Film_fMRI_zip, mode="wb")
  unzip(Rest_vs_Film_fMRI_zip, exdir=glue("{data_path}/Rest_vs_Film_fMRI/time_series_files/"))
  
  # Function to process the input file with 114 parcellations and aggregate to the broader 7 Yeo networks
  # input_filename: the .txt file containing the time-series data for 114 regions for a given subject ID, 
  # scanning condition, and scanning session number
  process_data_file <- function(input_filename) {
    sample_ID <- str_split(input_filename, "[.]")[[1]][1]
    scan_type <- str_split(input_filename, "[.]")[[1]][2]
    session_number <- str_split(input_filename, "[.]")[[1]][3]
    
    file_data <- read.table(glue("{data_path}/Rest_vs_Film_fMRI/time_series_files/{input_filename}"), sep=",") 
    colnames(file_data) = Yeo_parc_info$Full_Region_Name
    
    # Read in the time-series data for the 114 parcellated regions and join with Yeo parcellation lookup table
    data_in_full_networks <- file_data %>%
      mutate(timepoint = row_number()) %>%
      pivot_longer(cols=c(-timepoint),
                   names_to = "Full_Region_Name",
                   values_to = "BOLD_Signal") %>%
      left_join(., Yeo_parc_info) %>%
      mutate(Sample_ID = sample_ID,
             Scan_Type = scan_type,
             Session_Number = session_number) %>%
      mutate(Unique_ID = as.character(glue("{Sample_ID}_{Scan_Type}_{Session_Number}")))
    
    return(data_in_full_networks)
  }
  
  # Identify files to process
  fMRI_ts_files <- list.files(glue("{data_path}/Rest_vs_Film_fMRI/time_series_files/"), pattern="sub")
  
  # Process input files
  fMRI_ts_data_full <- fMRI_ts_files %>%
    purrr::map_df(~ process_data_file(.x)) %>%
    # Only take frames where all data is present
    filter(timepoint < 948)
  
  # Aggregate the 114 parcellation regions to the overarching 7 Yeo functional networks
  fMRI_ts_data_7_networks <- fMRI_ts_data_full %>%
    group_by(Sample_ID, Scan_Type, Session_Number, Unique_ID, Yeo_7_Networks_Bilateral, timepoint) %>%
    summarise(Mean_BOLD_Signal = mean(BOLD_Signal, na.rm=T)) %>%
    dplyr::rename("Node" = "Yeo_7_Networks_Bilateral",
                  "signal" = "Mean_BOLD_Signal") %>%
    group_by(Sample_ID, Scan_Type, Session_Number, Unique_ID, Node) %>%
    # z-score each time series
    mutate(signal_z = scale(signal)) 
  
  # Save time-series data to a feather file
  TAF::mkdir(glue("{data_path}/Rest_vs_Film_fMRI/processed_data/"))
  feather::write_feather(fMRI_ts_data_7_networks, glue("{data_path}/Rest_vs_Film_fMRI/processed_data/Rest_vs_Film_fMRI_TS_Yeo7.feather"))
  
} else {
  fMRI_ts_data_7_networks <- feather::read_feather(glue("{data_path}/Rest_vs_Film_fMRI/processed_data/Rest_vs_Film_fMRI_TS_Yeo7.feather"))
}

# Save a metadata file
fMRI_metadata <- fMRI_ts_data_7_networks %>%
  ungroup() %>%
  distinct(Unique_ID, Sample_ID, Scan_Type, Session_Number)
feather::write_feather(fMRI_metadata, glue("{data_path}/Rest_vs_Film_fMRI_metadata.feather"))

# Split by group
fMRI_TS_data_split  <- fMRI_ts_data_7_networks %>%
  # Reshape from long to wide such that columns are timepoints
  pivot_wider(id_cols = c("Unique_ID", "Node"),
              names_from = timepoint, values_from = signal_z) %>%
  dplyr::rename("Sample_ID" = "Unique_ID") %>%
  group_by(Sample_ID) %>%
  # Split into a list of dataframes per sample ID
  group_split()

# Write files to numpy arrays for pyspi processing
np <- import("numpy")
numpy_file_path <- as.character(glue("{data_path}/Rest_vs_Film_fMRI/numpy_files"))
TAF::mkdir(numpy_file_path)

1:length(fMRI_TS_data_split) %>%
  purrr::map(~ write_data_to_numpy(sample_df = fMRI_TS_data_split[.x][[1]]))

# Create a YAML file for these .npy files
ID_var <- "Sample_ID"
dim_order <- "ps"
yaml_file_base <- "sample.yaml"

cmd <- glue("Rscript {yaml_script_file} --data_dir {as.character(numpy_file_path)} --numpy_file_base .npy --ID_var {ID_var} --dim_order {dim_order} --yaml_file {yaml_file_base}")
system(cmd)