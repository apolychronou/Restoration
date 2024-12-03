from wildlife_datasets import datasets
from lib.experimentClass import ExpSetup, CombinedCallable, CombinedClass, get_all_metadata
from lib.myTrainer import mytest
from preparation.prepare_data import *
import os
import pandas as pd
import torch
import json
import re
from collections import defaultdict



def main():
    exper =  ExpSetup()
    datafr = []
    pattern = r"k=(\d+)-sf=(\d+)-sigma=([\d.]+)"

    for name in prepare_functions:
        print(f"Processing name: {name}")
        root_dir = os.path.join(exper.out_dir,name)
        if not os.path.exists(root_dir):
                print(f"No results for {name}")
                continue

        # Dictionary to group directories by (sf, sigma), and then by k
        dirs_by_sf_sigma = defaultdict(lambda: defaultdict(list))

        # Iterate over subdirectories
        for dir_name in os.listdir(root_dir):
            # print(dir_name)
            match = re.match(pattern, dir_name)
            if match:
                # Extract k, sf, and sigma from the directory name
                k, sf, sigma = int(match.group(1)), int(match.group(2)), float(match.group(3))
                
                # Group directories by (sf, sigma) and then by k
                dirs_by_sf_sigma[(sigma, sf)][k].append(os.path.join(root_dir, dir_name))

        averages_gt = {}
        averages_nl = {}
        averages_rc = {}

        for (sigma, sf), dirs_by_k in dirs_by_sf_sigma.items():
            for k, dirs in dirs_by_k.items():
                values_gt = []
                values_nl = []
                values_rc = []
                psnr_rc = []
                psnr_nl = []
                ssim_rc = []
                ssim_nl = []
                for dir_path in dirs:
                    file_path = os.path.join(dir_path, "accuracies.json")
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            # print(data)
                            values_gt.append(data["accuracy_gt"])
                            values_nl.append(data["accuracy_nl"])
                            values_rc.append(data["accuracy_rc"])
                        
                    file_path = os.path.join(dir_path, "metrics.json")
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            psnr_rc.append(data["PSNR"])
                            psnr_nl.append(data["PSNR no learning"])
                            ssim_rc.append(data["SSIM"])
                            ssim_nl.append(data["SSIM no learning"])

                
                # Calculate the average for this group
            averages_gt = sum(values_gt) / len(values_gt)
            averages_nl = sum(values_nl) / len(values_nl)
            averages_rc = sum(values_rc) / len(values_rc)
            average_psnr = sum(psnr_rc) / len(psnr_rc)
            average_psnr_nl = sum(psnr_nl) / len(psnr_nl)
            average_ssim = sum(ssim_rc) / len(ssim_rc)
            average_ssim_nl = sum(ssim_nl) / len(ssim_nl)
            datafr.append({
                "name"   : name,
                "sf"     : sf,
                "sigma"  : sigma,
                "avg_GT" : averages_gt,
                "avg_RC" : averages_rc,
                "avg_NL" : averages_nl,
                "avg_PSNR" : average_psnr,
                "avg_PSNR_NL" : average_psnr_nl,
                "avg_SSIM" : average_ssim,
                "avg_SSIM_NL" : average_ssim_nl
            })

    df = pd.DataFrame(datafr)
    df = df.sort_values(by=["name", "sf"]).reset_index(drop=True)

    print(df)
    df.to_csv(os.path.join(exper.out_dir,"averages.csv"), index=False)
    with open(os.path.join(exper.out_dir,'mytable.tex'), 'w') as tf:
        tf.write(df.to_latex())

if __name__ == '__main__':
    main()