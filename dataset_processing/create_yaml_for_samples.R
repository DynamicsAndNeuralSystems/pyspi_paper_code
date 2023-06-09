library(tidyverse)
library(argparse)
library(tools)
library(feather)
library(glue)

parser <- ArgumentParser(description='Automatically generate sample YAML file for pyspi-distribute.')

parser$add_argument("--data_dir", help="Directory containing samples' .npy files.")
parser$add_argument("--ID_var", help="OPTIONAL: ID variable in CSV metadata file.")
parser$add_argument("--numpy_file_base", help="OPTIONAL: extension for input numpy files.", default=".npy")
parser$add_argument("--label_vars", help="OPTIONAL: columns in metadata to use as labels for YAML.", nargs="?")
parser$add_argument("--sample_metadata_file", help="OPTIONAL: .Rds or .feather file containing sample metadata info.")
parser$add_argument("--dim_order", help="OPTIONAL: orientation of observations and time points in data. Default is ps, meaning time points are rows and samples are columns.", default="ps")
parser$add_argument("--overwrite", help="Should sample.yaml be overwritten if it already exists? Default is F.",
                    action="store_true", default=FALSE)
parser$add_argument("--yaml_file", help="OPTIONAL: Name of output sample YAML file. Default is sample.yaml.",
                    default="sample.yaml")

# Parse arguments
args <- parser$parse_args()
data_dir <- args$data_dir
ID_var <- args$ID_var
numpy_file_base <- args$numpy_file_base
dim_order <- args$dim_order
sample_metadata_file <- args$sample_metadata_file
label_vars <- args$label_vars
overwrite <- args$overwrite
yaml_file_base <- args$yaml_file

if (!endsWith(data_dir, '/')) {
  data_dir <- paste0(data_dir, "/")
}

npy_files <- list.files(data_dir, pattern="*.npy")
yaml_file <- paste0(data_dir, yaml_file_base)

cat("\nYAML output:", yaml_file, "\n")

if (!is.null(sample_metadata_file)) {
  if (endsWith(sample_metadata_file, '.Rds')) {
    sample_metadata <- readRDS(sample_metadata_file)
  } else {
    sample_metadata <- feather::read_feather(sample_metadata_file)
  }
}

if (!is.null(label_vars)) {
  sample_metadata <- sample_metadata %>%
    dplyr::rename("ID_var" = glue("{ID_var}")) 
}

if (!file.exists(yaml_file) | overwrite) {
  cat("\nNow creating sample.yaml\n")
  file.create(yaml_file)
  yaml_string <- "- {file: %s, name: %s, dim_order: %s, labels: [%s] }\n"
  for (npy in npy_files) {
    sample_ID <- gsub(numpy_file_base, "", npy)
    if (!is.null(sample_metadata_file) & !is.null(label_vars)) {
      sample_data <- sample_metadata %>%
        dplyr::filter(ID_var == sample_ID) %>%
        dplyr::select(all_of(label_vars))
      sample_data_vector <- paste(as.vector(sample_data[1,]), collapse=",")
    } else{
      sample_data_vector <- ""
    }
    
    sample_string <- sprintf(yaml_string, 
                             paste0(data_dir, npy),
                             sample_ID,
                             dim_order,
                             sample_data_vector)
    write_file(sample_string, yaml_file, append=T)
  }
}