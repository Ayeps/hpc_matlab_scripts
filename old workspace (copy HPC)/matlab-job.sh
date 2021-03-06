#!/bin/bash # Load modules directive
#. /etc/profile.d/modules.s

# Reserve N workers
#$ -pe smp 12
#

# Delete previous outputs generated
#rm /homedtic/fwilhelmi/workspace/output/*

# Copy sources to the SSD:

# First, make sure to delete previous versions of the sources:
# ------------------------------------------------------------
if [ -d /scratch/QL_Experiments ]; then
        rm -Rf /scratch/QL_Experiments
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/QL_Experiments
mkdir /scratch/QL_Experiments/data
mkdir /scratch/QL_Experiments/error
mkdir /scratch/QL_Experiments/script
mkdir /scratch/QL_Experiments/out

# Third, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/fwilhelmi/workspace/script/* /scratch/QL_Experiments/script

# Fourth, prepare the submission parameters:
# Remember SGE options are marked up with '#$':
# ---------------------------------------------
# Requested resources:
#
# Simulation name
# ----------------
#$ -N "QL-Testing_parameters"
#
# Shell
# -----
#$ -S /bin/bash
#
# Output and error files go on the user's home:
# -------------------------------------------------
#$ -o /homedtic/fwilhelmi/workspace/output/output.out
#$ -e /homedtic/fwilhelmi/workspace/output/error.err
#

# Start script
# --------------------------------
#
printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME\n"
printf "Starting at `date`\n"
printf "Calling Matlab now\n"
# Execute the script
/soft/MATLAB/R2015b/bin/matlab -nosplash -nodesktop -nodisplay -r "run /scratch/QL_Experiments/script/Experiment_2_QL_tune_parameters.m"
# Copy data back, if any
printf "Matlab processing done. Moving data back\n"
# cp -rf /scratch/QL_Experiments/out/ql_experiments.out /homedtic/fwilhelmi/workspace/output
printf "Job done. Ending at `date`\n"
