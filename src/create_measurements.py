from wildlife_datasets import datasets
from lib.experimentClass import ExpSetup, CombinedCallable, CombinedClass
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
from deepinv.models import DRUNet
from deepinv.optim.dpir import get_DPIR_params
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import PnP
from deepinv.optim.data_fidelity import L2
from deepinv.physics.generator import SigmaGenerator
import json
import matplotlib.pyplot as plt
import copy


def main():
    noise_levels    =   [0.01, 0.03]
    test_sf         =   [2, 4]
    k_num           =   8
    seed            =   666
    img_size        =   [224, 224]
    device          =   dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    num_workers     =   10 if torch.cuda.is_available() else 0
    batch_size      =   11 if torch.cuda.is_available() else 1
    exper           =   ExpSetup()


    np.random.seed(seed=seed)
    for name, prepare_fn in prepare_functions.items():
        print(f"name: {name} - function: {prepare_fn}")

        if not os.path.exists(f"{exper.data_dir}/{name}"):
            try:
                prepare_fn(root=f"{exper.datasets_dir}/{name}", new_root=f"{exper.data_dir}/{name}", size=img_size)
            except Exception as e:
                print(e)
                continue
        metadata = pd.read_csv(f'{exper.data_dir}/{name}/annotations.csv', index_col=0)
        splitter = splits.ClosedSetSplit(0.8, identity_skip='unknown', seed=seed)
        idx_train, idx_test = splitter.split(metadata)[0]

        metadata.loc[metadata.index[idx_train], 'split'] = 'train'
        metadata.loc[metadata.index[idx_test], 'split'] = 'test'

        print(f'{exper.work_dir}/kernels')
        kernels = hdf5storage.loadmat(os.path.join(f'{exper.work_dir}/kernels', 'kernels_12.mat'))['kernels']

        denoiser_pt=DRUNet(device=device, pretrained='download')
        prior = PnP(denoiser_pt)

        data_fidelity = L2()
        early_stop = False

        for sf in test_sf:

            transform = T.Compose([
            T.ToTensor(),
            ModCrop(sf)
            ])

            for noise_level_img in noise_levels:

                sigma_generator = SigmaGenerator(sigma_min=noise_level_img, sigma_max=noise_level_img)

                for k_idx in range(k_num):
                    k = kernels[0, k_idx].astype(np.float32)
                    k = torch.from_numpy(k)
                    k = k.unsqueeze(0).unsqueeze(0)

                    physics = Downsampling(filter = k, img_size=(3,)+tuple(img_size), 
                            noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                            factor=sf)
                    
                    if device.type == 'cuda':
                        physics_gpu = Downsampling(filter = k, img_size=(3,)+tuple(img_size), 
                            noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                            factor=sf, device=device)

                    device1 = next(physics.parameters()).device
                    device2 = next(physics_gpu.parameters()).device

                    sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level_img)
                    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
                    
                    trans_degrade = CombinedCallable(transform, lambda x: torch.unsqueeze(x, dim=0), physics, lambda x: torch.squeeze(x, dim=0), generator=sigma_generator)

                    dataset = WildlifeDataset(
                        metadata = metadata, 
                        root = f'{exper.data_dir}/{name}',
                        split = SplitMetadata('split', 'test'),
                        transform=transform
                        )
                    
                    degraded_dataset = WildlifeDataset(
                        metadata = metadata, 
                        root = f'{exper.data_dir}/{name}',
                        split = SplitMetadata('split', 'test'),
                        transform=trans_degrade
                        )
                    
                    data_pair = CombinedClass(dataset, degraded_dataset)

                    dl_data_pair = DataLoader(data_pair, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers,
                                                drop_last=False, pin_memory=True)
                    
                    model = optim_builder(
                        iteration="HQS",
                        prior=prior,
                        data_fidelity=data_fidelity,
                        early_stop=early_stop,
                        max_iter=max_iter,
                        verbose=True,
                        params_algo=params_algo,
                    )

                    model.eval()


                    trainer = mytest(
                        test_dataloader=dl_data_pair,
                        model=model,
                        metrics=[dinv.metric.PSNR(), dinv.metric.SSIM()],
                        device=device,
                        show_progress_bar=True,
                        display_losses_eval=True,
                        save_folder=f"{exper.out_dir}/{name}/k={k_idx+1}-sf={sf}-sigma={noise_level_img:.2f}",
                        plot_convergence_metrics=False,
                        plot_images=False,
                        verbose=True,
                        online_measurements=False,
                        physics=physics_gpu,
                        df=metadata[metadata['split'] == 'test'],
                        no_learning_method='interpolation'
                    )

                    if k_idx == 0:
                        cum_metrics = copy.deepcopy(trainer)
                    else:
                        for key, value in trainer.items():
                            cum_metrics[key] += value

                    with open(f"{exper.out_dir}/{name}/k={k_idx+1}-sf={sf}-sigma={noise_level_img:.2f}/metrics.json", "w") as json_file:
                        json.dump(trainer, json_file)
                

                    del dl_data_pair, trainer, physics_gpu, model, dataset

                cum_metrics = {key: value / k_num for key, value in cum_metrics.items()}
                with open(f"{exper.out_dir}/{name}/k-all-sf={sf}-sigma={noise_level_img:.2}-metrics.json", "w") as json_file:
                    json.dump(cum_metrics, json_file) 

                

                
if __name__ == '__main__':
    main()