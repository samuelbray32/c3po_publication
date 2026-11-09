partition = 3
gpu_to_use = 9

from pathlib import Path
output_dir_base = Path("/stelmo/sam/c3po_dandi_training_v2")
output_dir_base.mkdir(parents=True, exist_ok=True)

import pandas as pd
file_path = "/home/sambray/Documents/c3po/tutorial/c3po_dandi_candidate_datasets_v2.csv"
dataset_df = pd.read_csv(file_path)
# partitions = [
#     (0,8),
#     (8,16),
#     (16,23),
#     (23,31)
# ]
# dataset_df = dataset_df.iloc[partitions[partition][0]:partitions[partition][1]]
import numpy as np
partitions = np.split(dataset_df, 4)
dataset_df = partitions[partition]
import os
import subprocess
import sys

def main():
    env = os.environ.copy()
    for i, row in dataset_df.iterrows():
        dandiset_id = row['dandiset_id']
        dandiset_id = f"{dandiset_id:06d}"
        dandi_path = row['dandi_path']
        dandi_instance = "dandi"
        print(f"Processing {dandiset_id}...")
        output_dir = output_dir_base/dandiset_id
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
            [
                sys.executable,
                "/home/sambray/Documents/c3po/tutorial/train_from_dandi.py",
                "--dandiset_id",
                str(dandiset_id),
                "--dandi_path",
                str(dandi_path),
                "--dandi_instance",
                str(dandi_instance),
                "--output_dir",
                str(output_dir),
                "--cuda-device",
                str(gpu_to_use),
            ],
            env=env,
            check=True,
        )
        except Exception as e:
            print(f"Error processing {dandiset_id}: {e}")

if __name__ == "__main__":
    main()