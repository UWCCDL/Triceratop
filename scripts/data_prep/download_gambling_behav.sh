#!/bin/bash

cd /projects/HCP/old

while read subject; do
    echo "Processing subject: $subject"
    
    # Loop through sessions (LR and RL phase encoding directions)
    for session in LR RL; do
        echo "  Processing session: $session"
        
        # Create destination directory if it doesn't exist
        mkdir -p /projects/HCP/Gambling/${subject}/behav/${session}
        
        # Download the entire EPRIME directory (contains all behavioral data)
        aws s3 cp --recursive \
            s3://hcp-openaccess/HCP_1200/${subject}/unprocessed/3T/tfMRI_GAMBLING_${session}/LINKED_DATA/EPRIME \
            /projects/HCP/Gambling/${subject}/behav/${session}
        
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "  Successfully downloaded EPRIME data for ${subject}, session ${session}"
        else
            echo "  ERROR: Failed to download EPRIME data for ${subject}, session ${session}"
        fi
    done
done < HCP_subids_1041.txt


#supplementary code for locating data :/
aws s3 ls s3://hcp-openaccess/HCP/${subject}/ --recursive | grep -i gambling

aws s3 ls s3://hcp-openaccess/HCP/100206/ --recursive | grep -i gambling

aws s3 ls s3://hcp-openaccess/ --recursive | grep 100206 | grep -i gambling

aws s3 cp --recursive \
    s3://hcp-openaccess/HCP_1200/${subject}/unprocessed/3T/tfMRI_GAMBLING_${session}/LINKED_DATA/EPRIME \
    /projects/HCP/Gambling/${subject}/behav/${session}