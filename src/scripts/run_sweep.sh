#!/bin/bash

#$ -M yzhang46@nd.edu
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -j y

#$ -pe smp 2             
#$ -q long          

#$ -N casa
#$ -q gpu@@csecri    # gpu@@csecri-p100, gpu@@csecri-titanxp 
#$ -l gpu_card=1


export PROJ_PATH="/afs/crc.nd.edu/user/y/yzhang46/_"


if [ "$USER" == "yzhang46" ]; then

        # Env and Requirements Setup
        cd $PROJ_PATH

        module load pytorch
        module load python 
        

        # echo -e "\n>>> Installing Python requirements\n"
        # pip3 install --user -r requirements.txt
        # echo -e "\n>>> Done installing Python requirements\n"

        # echo -e "\n>>> Logging in to W&B\n"
        # . ./scripts/wandb_login.sh
        # echo -e "\n>>> Done logging in to W&B\n"

        cd "$PROJ_PATH/src"

else

        echo -e "Job script outside CRC GPU-env not implemented."

fi


# CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES="${SGE_HGR_gpu_card// /,}"
if [ -z ${SGE_HGR_gpu_card+x} ]; then 
        SGE_HGR_gpu_card=-1
fi
echo -e "Assigned GPU(s): ${SGE_HGR_gpu_card}\n"
echo -e "Starting Experiment =)"
echo -e "=-=-=-=-=-=-=-=-=-=-=-=-=\n"

wandb agent charzhar/WANDB_PROJECT_NAME/SWEEP_ID --count 5
