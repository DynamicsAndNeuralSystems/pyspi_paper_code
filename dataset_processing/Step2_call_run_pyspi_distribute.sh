#!/usr/bin/env bash

export github_dir=/path/to/your/github/on/cluster

# Clone pyspi-distribute repo onto the cluster with
# git clone git@github.com:anniegbryant/pyspi-distribute .
export pyspi_distribute_script_dir=${github_dir}/pyspi-distribute/
export pyspi_config=config_used_for_pyspi_paper.yaml # Feel free to swap out with another config file to try other SPIs
export pkl_file=calc.pkl
export template_pbs_file=${pyspi_script_dir}/template.pbs
export email=your_email@email.com
export conda_env=pyspi # Specify a conda environment into which you've installed pyspi -- could just be base if you installed into your base environment
export queue=insert_queue_name_here # Check with your HPC -- this is the name of the queue you want to submit to
export pyspi_walltime_hrs=6 # Specify a walltime in hours
export pyspi_ncpus=2 # Specify number of CPUs to request per sample to process
export pyspi_mem=20 # Specify memory (in GB) to request per sample to process
export sample_yaml=sample.yaml # Specify the name of the sample yaml file for the datasets you want to process
export data_path=/path/to/your/data/on/cluster # This is the path to the directory containing data for BasicMotions, SelfRegulationSCP1, and Rest_vs_Film_fMRI on the cluster

# Activate given conda env
conda activate ${conda_env}

##########################################################################################################
# Iterate over datasets and submit all parallelized pyspi jobs with qsub via distribute_jobs.py
for dataset_ID in BasicMotions SelfRegulationSCP1 Rest_vs_Film_fMRI; do 
    cmd="python $pyspi_script_dir/distribute_jobs.py \
    --data_dir ${data_path}/${dataset_ID}/numpy_files/ \
    --calc_file_name $pkl_file \
    --compute_file $pyspi_script_dir/pyspi_compute.py \
    --template_pbs_file $template_pbs_file \
    --pyspi_config $pyspi_config \
    --sample_yaml ${data_path}/${dataset_ID}/numpy_files/${sample_yaml} \
    --pbs_notify a \
    --email $email \
    --conda_env $conda_env \
    --queue $queue \
    --walltime_hrs $pyspi_walltime_hrs \
    --cpu $pyspi_ncpus \
    --mem $pyspi_mem \
    --table_only"
    echo $cmd
    $cmd
done