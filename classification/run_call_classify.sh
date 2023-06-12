#!/usr/bin/env bash 

export data_path=/path/to/data/on/cluster/ # Change this to the path to the data on the cluster
export conda_env=my_conda_env # Specify which conda environment you want to use
export queue=insert_queue_name_here # Check with your HPC -- this is the name of the queue you want to submit to
export ncpus=4 # Specify number of CPUs to request per sample to process
export mem=40 # Specify memory (in GB) to request per sample to process
export mpi_procs=4 # Specify number of MPI processes to request per sample to process
export walltime_hrs=40 # Specify a walltime in hours

# Run call_classify.pbs for each of the following values of dataset_ID: BasicMotions, SelfRegulationSCP1, Rest_vs_Film_fMRI
for dataset_ID in BasicMotions SelfRegulationSCP1 Rest_vs_Film_fMRI
do
    # Run call_classify.pbs
    cmd="qsub -v data_path=$data_path,dataset_ID=$dataset_ID,conda_env=$conda_env -q $queue -l select=1:ncpus${ncpus}:mem=${mem}GB:mpiprocs=${mpi_procs} -l walltime=${walltime_hrs}:00:00 -M your_email@email.com -N ${dataset_ID}_classify -o /cluster/output/path/classify_${dataset_ID}_out.txt call_classify.pbs"
    echo $cmd
    $cmd
done