# Code for "Unifying pairwise interactions in complex dynamics"

[![DOI](https://zenodo.org/badge/651395505.svg)](https://zenodo.org/badge/latestdoi/651395505)

Readme and code in this repo were compiled by @olivercliff (originally published in [https://github.com/olivercliff/pyspi-paper](https://github.com/olivercliff/pyspi-paper)) with additions by @anniegbryant.

This repository illustrates how the figures in the paper, "Unifying pairwise interactions in complex dynamics", were created.

We provide both precomputed CSV files, with which to recreate the figures, as well as scripts to generate these CSVs from scratch.

## Download pyspi and create an environment

First, download `pyspi` and create a conda environment to install the package [as per the documentation](https://pyspi-toolkit.readthedocs.io/en/latest/).
In linux, this involves the following steps from a terminal (in your desired directory):

```
git clone git@github.com:DynamicsAndNeuralSystems/pyspi.git
cd pyspi
conda create -n pyspi python=3.9
conda activate pyspi
```

> **NOTE:** If using the legacy `pynats` branch, please create the conda environment with `python=3.6.7`.

You will also need to download and install `octave`; [follow the instructions here](https://octave.org/download).

## Switch to the `pynats` branch and install

The `pynats` branch was retroactively added to the `pyspi` repository as legacy code that was used to generate the results from the main paper.
If you would like to replicate the results from the paper as closely as possible, we would recommend checking out the `pynats` branch, which is [also available as a release](https://github.com/DynamicsAndNeuralSystems/pyspi/releases/tag/pynats-v0.1).
However, given this work was not computed via a container, such as docker, the results may vary slightly from those originally reported.

In the `pyspi` folder and environment that were created above, checkout and install the `pynats` branch (this will take a while):

```
git checkout pynats
pip install .
```

# Figure 2: Hierarchical clustering for SPI performance on 1053 MTS dataset

Because of the large amount of processing required to generate this figure from scratch, we have provided pre-computed CSV files (in the `data` directory) from which you can easily regenerate the components of Figure 2 using the [`Generate_Figure2_Visuals.ipynb`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main//Generate_Figure2_Visuals.ipynb) notebook.

In order to re-compute the CSV files from the raw MTS data, we have provided the following script:
[`process_mts_database.py`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/process_mts_database.py), which computes all SPIs for each of the 1053 MTS datasets in the database.

Once this script has been executed, re-run the `Generate_Figure2_Visuals.ipynb` notebook, pointing to the location of the new CSV files to regenerate the figures.

> **NOTE:** The script `process_mts_database.py` was computed on a cluster, where each dataset was evaluated separately. If running this script locally, it will likely take several months to complete (and so we would advise using the [pyspi distribute](https://github.com/DynamicsAndNeuralSystems/pyspi-distribute) workflow to recompute all SPIs on all 1053 datasets, if necessary).

# Figure 3: Classification case study with three example datasets

This repository also includes all the code you need to replicate our classification case study, separated into four parts:

1. [Downloading the data](#step-1-downloading-the-data)
2. [Computing SPIs with pyspi](#step-2-running-pyspi-distribute-on-a-cluster)*
3. [Fitting linear SVM classifiers](#step-3-fitting-linear-svm-classifiers)*
4. [Visualizing results as in the manuscript](#step-4-visualizing-results-as-in-the-manuscript)

\*The code is structured such that these steps are run as distributed jobs on a high-performance computing (HPC) cluster given the size of the datasets.
If you do not have access to such a cluster, the code can be adapted to run locally -- however, note that it can take a very long time, depending on your computer specs.

> **NOTE:** [`dataset_processing/parse_datasets_for_pyspi.R`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/parse_datasets_for_pyspi.R) and [`dataset_processing/parse_datasets_for_pyspi.R`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/call_run_pyspi_distribute.sh) assume you have `pyspi-distribute` installed, and will need to be modified accordingly if not.

If you would like to skip ahead to visualizing pre-computed results, you may download pre-computed `.feather` files from [zenodo](https://doi.org/10.5281/zenodo.8216295).
The statistics underlying the visualizations can also be found in the following files:

- `data/All_combined_SPI_stats_for_Fig3_histogram.csv`
- `data/Individual_SPI_stats_for_Fig3_histogram.csv`
- `data/SPI_stats_for_Fig3_violin_plots.csv`

## Step 1: Downloading the data

All data included in our classification analysis is freely available to the public from the following sources:

1. Smartwatch activity dataset: [here](https://timeseriesclassification.com/description.php?Dataset=BasicMotions)
2. EEG state dataset: [here](https://timeseriesclassification.com/description.php?Dataset=SelfRegulationSCP1)
3. fMRI film dataset: [here](https://figshare.com/articles/dataset/Resting-state_and_movie-watching_data/12971162)

We have included an R script to automatically download and parse the data from these scripts for you, located at [`dataset_processing/parse_datasets_for_pyspi.R`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/parse_datasets_for_pyspi.R).
Make sure to update [line 12](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/cf569e4097b4570eaf7dedc2b2f51bade745c102/dataset_processing/parse_datasets_for_pyspi.R#L12) to the path where you have cloned `pyspi-distribute` if you did so.
You might also wish to change the path to where you want the data to be downloaded and processed; the default is the `data/` folder in this repository, but you can modify this on line 11.

This script will automatically download and extract the data for the three datasets, combine the time-series data into `feather` files, and output individual samples' multivariate time series (MTS) matrices as `NumPy` binary files (.npy).
This is the file format supplied to `pyspi-distribute` via the `sample.yaml` file that the script automatically generates.

If this step runs successfully, you should have the following three folders within your `data/` folder in your cloned version of this repo:

1. `BasicMotions/` (smartwatch activity dataset)
2. `SelfRegulationSCP1/` (EEG state dataset)
3. `Rest_vs_Film_fMRI/` (fMRI film dataset)

Within each of these folders, you should have the following subfolders:

1. `arff_files/` OR `time_series_files/`
2. `numpy_files/`
3. `processed_data/`

The `numpy_files/` subfolder should include one `.npy` file for each sample in the given MTS and one `sample.yaml` configuration file for `pyspi-distribute`.

## Step 2: Running pyspi-distribute on a cluster

Once your data is downloaded and prepped from [Step 1](#step-1-downloading-the-data), you can run `pyspi-distribute` on the cluster. If you wish to use the script provided in this repo (`call_run_pyspi_distribute.sh`), note that you should modify lines 3-18 to supply your data path on the cluster, the name of your conda environment where `pyspi` is installed, and PBS job specifications such as memory and walltime to request.
Some sensible defaults for resource requests are provided as a starting point.

After updating those lines, you can run `call_run_pyspi_distribute.sh` from your terminal on an HPC cluster with the following:

```
bash dataset_processing/call_run_pyspi_distribute.sh
```

Note that by default, this script submits all PBS jobs to be queued and scheduled using `qsub` at once; if you prefer to submit `pyspi` jobs one case study dataset at a time, you can update line 25 to comment out individual datasets.

Once all of the MTS have been processed with `pyspi-distribute`, you can combine the outputs from individual jobs with [`dataset_processing/merge_psypi_data.py`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/merge_psypi_data.py). If you have your data stored somewhere other than within the `data/` folder in this repository, you should update the paths on lines 137, 158, and 179 accordingly. You can run this python script with the following:

```
python3 dataset_processing/merge_pyspi_data.py
```

For the classification case study in our paper, we examined the 237 SPIs that were available at the time of manuscript drafting (which can be found in [`data/SPI_info.csv`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/data/SPI_info.csv)).
Of these, for each of the three classification datasets, we filtered out any SPIs that yielded constant values (i.e., returned the same value for every pair in the MTS across all samples) and/or could not be computed for all samples, presumably due to numerical issues. These quality control steps are applied in [`dataset_processing/prep_datasets_for_classification.py`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/prep_datasets_for_classification.py), which can be run with the following:

```
python3 dataset_processing/prep_datasets_for_classification.py
```

This will prepare DataFrames in `.feather` files to fit linear SVMs and classify groups as implemented in our paper.

## Step 3: Fitting linear SVM classifiers

To evaluate how well each SPI could separate between classes in the three datasets, we fit linear support vector machine (SVM) classifiers using the `sklearn.svm.SVC()` function with default parameters for each SPI.
For the smartwatch activity and EEG state datasets, we performed stratified resampling 30 times with the same proportions applied in [Ruiz et al. (2021)](https://doi.org/10.1007/s10618-020-00727-3).
For the film fMRI dataset, since there were 29 individuals in the dataset, we performed leave-one-individual-out cross-validation (LOOCV) such that for each of 29 iterations, the SVM was trained on the resting and film-viewing fMRI scans from 28 individuals and evaluated on an unseen test set of the left-out individual's rest and film-viewing fMRI scans.
In each case, we fit null models by shuffling the class labels and fitting classifiers to the shuffled data 100x per SPI to yield 100 null accuracy values for SPI; we opted to pool null values together across SPIs to create a joint null distribution to evaluate statistical significance.
We also evaluated the performance of all SPIs combined into one SVM classifier for problem with the same resampling approaches.

This classification step can be reproduced by running [`dataset_processing/run_call_classify.sh`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/run_call_classify.sh).
Here, you should modify arguments in lines 3-9 to specify where you have the data from this project stored, your conda environment, and parameters for your qsub job submission such as walltime and memory to request.

Once you have modified [`dataset_processing/run_call_classify.sh`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/run_call_classify.sh) as appropriate, you can run the script as follows:

```
bash classification/run_call_classify.sh
```

This `bash` script submits the PBS script [`dataset_processing/call_classify.pbs`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/dataset_processing/call_classify.pbs) for each of the three datasets to be processed in parallel as three separate PBS jobs.
Each job will run [`classification/classify.py`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/classification/classify.py) on the given dataset, fitting SVMs with resampling to each SPI separately as well as to all SPIs together.
For each dataset, this will output four `.feather` files to the data directory you specify in line 3 of [`classification/run_call_classify.sh`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/classification/run_call_classify.sh):

1. `{dataset_ID}_main_SPI_wise_acc.feather`: The classification accuracy for each resample for SVMs fit to each SPI individually
2. `{dataset_ID}_null_SPI_wise_acc.feather`: The null classification accuracy for each resample for SVMs fit to each SPI individually
3. `{dataset_ID}_main_full_acc.feather`: The classification accuracy for each resample with all SPIs combined into one SVM
4. `{dataset_ID}_null_full_acc.feather`: The null classification accuracy for each resample for all SPIs combined into one SVM

## Step 4: Visualizing results as in the manuscript

All result visualizations in the classification case study analysis were generated using R.
To replicate these figures, you can run the code chunks in the R Markdown [`Generate_Figure3_Visuals.Rmd`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/Generate_Figure3_Visuals.Rmd).
The output from this Markdown document is included as a static HTML docoment that you can view, [`Generate_Figure3_Visuals.html`](https://github.com/DynamicsAndNeuralSystems/pyspi_paper_classification/blob/main/Generate_Figure3_Visuals.html).

Note that the script does not automatically save output figures, so users may manually save visualizations using the [`ggsave()`](https://ggplot2.tidyverse.org/reference/ggsave.html) function.
