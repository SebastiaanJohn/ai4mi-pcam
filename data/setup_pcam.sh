#!/bin/bash

# This script is used to download the PCam dataset and extract the images and
# annotations. When running this script, your cwd should be the root of the
# project. Run this script from the command line as `./data/setup_pcam.sh`.

raw_dir="./data/processed/pcam"

# Download the raw dataset annotations.
wget 'https://drive.google.com/uc?export=download&id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO&confirm=t' -O "${raw_dir}/camelyonpatch_level_2_split_test_y.h5.gz"
wget 'https://drive.google.com/uc?export=download&id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO&confirm=t' -O "${raw_dir}/camelyonpatch_level_2_split_valid_y.h5.gz"
wget 'https://drive.google.com/uc?export=download&id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG&confirm=t' -O "${raw_dir}/camelyonpatch_level_2_split_train_y.h5.gz"

# Download the raw dataset images.
wget 'https://drive.google.com/uc?export=download&id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_&confirm=t' -O "${raw_dir}/camelyonpatch_level_2_split_test_x.h5.gz"
wget 'https://drive.google.com/uc?export=download&id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3&confirm=t' -O "${raw_dir}/camelyonpatch_level_2_split_valid_x.h5.gz"
wget 'https://drive.google.com/uc?export=download&id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2&confirm=t' -O "${raw_dir}/camelyonpatch_level_2_split_train_x.h5.gz"

# Extract the dataset.
for source_file in "$raw_dir"/*.gz; do
    dest_file="${source_file%.gz}"
    echo "Extracting ${dest_file}..."
    # if pv is installed, use pv to show progress. otherwise, just use gunzip.
    if pv --version >/dev/null 2>&1; then
        pv "${source_file}" | gunzip > "${dest_file}"
        rm "${source_file}"
    else
        gunzip "${source_file}"
    fi
done

# Run the script to pre-process the dataset.
echo "Preprocessing the dataset..."
# TODO Create Python script to pre-process the dataset.

# Print confirmation message.
echo -e "\n😄🎉🌟💃🥳🤩🥂👑🤗💥 >>> PCam setup script finished <<< 🌺🌈🕺🎈👍🏆💯🥇🚀👌"
