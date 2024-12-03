from wildlife_datasets import datasets
from lib.experimentClass import ExpSetup, CombinedCallable, CombinedClass, get_all_metadata
from lib.utils_image import ModCrop
from lib.myTrainer import mytest
from preparation.prepare_data import *
import os
import pandas as pd
from wildlife_datasets import splits
from wildlife_tools.data.split import SplitMetadata
import hdf5storage
import torch
from torch.utils.data import DataLoader
from deepinv.physics import Downsampling
import deepinv as dinv
import json
import matplotlib.pyplot as plt
from wildlife_tools.features import DeepFeatures
import timm
from wildlife_tools.similarity.cosine import CosineSimilarity
from wildlife_tools.inference.classifier import KnnClassifier



def main():
    seed            =   666
    device          =   dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    num_workers     =   10 if torch.cuda.is_available() else 0
    batch_size      =   512 if torch.cuda.is_available() else 1
    exper           =   ExpSetup()
    megaDescriptor  =   timm.create_model("hf-hub:BVRA/MegaDescriptor-L-224", pretrained=True)

    np.random.seed(seed=seed)
    for name in prepare_functions:
        print(f"Processing name: {name}")
        
        if not os.path.exists(os.path.join(exper.out_dir,name)):
                print(f"No results for {name}")
                continue
        if not os.path.exists(os.path.join(exper.out_dir, name)):
                print(f"No database for {name}")

        metadata_db = pd.read_csv(f'{exper.data_dir}/{name}/annotations.csv', index_col=0)
        splitter = splits.ClosedSetSplit(0.8, identity_skip='unknown', seed=seed)
        idx_train, idx_test = splitter.split(metadata_db)[0]
        dataset_database = WildlifeDataset(metadata_db.loc[idx_train], os.path.join(exper.data_dir, name), transform=T.ToTensor())
        
        with torch.no_grad():
            extractor = DeepFeatures(megaDescriptor, device=device, batch_size=batch_size, num_workers=num_workers)
            database = extractor(dataset_database)
            matcher = CosineSimilarity()

        
        for dirpath, dirnames, _ in os.walk(os.path.join(exper.out_dir,name)):
            for dir in dirnames:
                data_folder = os.path.join(dirpath,dir)
                metadata_qr = get_all_metadata(data_folder)
                accuracies = {}
                for key, value in metadata_qr.items():                    
                    dataset_query = WildlifeDataset(value, data_folder, transform=T.ToTensor())

                    with torch.no_grad():
                        similarity = matcher(query=extractor(dataset_query), database=database)
                        knnClass = KnnClassifier(k=10, database_labels=dataset_database.labels_string)
                        preds = knnClass(similarity['cosine'])
                        acc = sum(preds == dataset_query.labels_string) / len(preds)
                    
                    accuracies[f"accuracy_{key}"] = acc
                    
                    with open(os.path.join(data_folder,"accuracies-L.json"), "w") as json_file:
                        json.dump(accuracies, json_file)
            dirnames.clear()







if __name__ == '__main__':
    main()