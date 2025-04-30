mkdir -p /projects/HCP/Gambling/all
touch subject_list.txt

while read subject; do
    for session in RL LR; do
        if [ -e /projects/HCP/Gambling/${subject}/behav/${session}/${subject}_*.txt ]; then
        	cp /projects/HCP/Gambling/${subject}/behav/${session}/${subject}_*.txt /projects/HCP/Gambling/all/
            grep -q "^${subject}$" subject_list.txt || echo "$subject" >> subject_list.txt
        fi
    done
done < HCP_subids_1041.txt

