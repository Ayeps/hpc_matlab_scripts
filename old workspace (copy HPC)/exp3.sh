#!/bin/bash # Load modules directive
#. /etc/profile.d/modules.s

# Reserve N workers
#$ -pe smp 8
#

# Delete previous outputs generated
#rm /homedtic/fwilhelmi/workspace/output/*

# Copy sources to the SSD:

# First, make sure to delete previous versions of the sources:
# ------------------------------------------------------------
if [ -d /scratch/QL_Experiments ]; then
        rm -Rf /scratch/EXP3_Experiments
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/EXP3_Experiments
mkdir /scratch/EXP3_Experiments/data
mkdir /scratch/EXP3_Experiments/error
mkdir /scratch/EXP3_Experiments/script
mkdir /scratch/EXP3_Experiments/out

# Third, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/fwilhelmi/workspace/script/* /scratch/EXP3_Experiments/script

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
#$ -o /homedtic/fwilhelmi/workspace/output/exp3_output.out
#$ -e /homedtic/fwilhelmi/workspace/output/exp3_error.err
#

# Start script
# --------------------------------
#
printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME\n"
printf "Starting at `date`\n"
printf "Calling Matlab now\n"
# Execute the script
/soft/MATLAB/R2015b/bin/matlab -nosplash -nodesktop -nodisplay -r "run /scratch/EXP3_Experiments/script/EXP3_EXPERIMENT_1.m"
# Copy data back, if any
printf "Matlab processing done. Moving data back\n"
