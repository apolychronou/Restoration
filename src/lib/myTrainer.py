from deepinv.utils import plot, plot_curves
import torch.nn.functional as F
from deepinv import Trainer
from deepinv.utils.plotting import prepare_images, preprocess_img
import wandb
from pathlib import Path
from torchvision.utils import save_image
import os
from tqdm import tqdm
import torch
import deepinv as dinv


class myTrainer(Trainer):

    def __init__(self, *args, df=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.df_gt = self.df.copy()
        self.df_ms = self.df.copy()
        self.df_rc = self.df.copy()
        self.df_nl = self.df.copy()
        self.df_gt['path'] = self.df_gt['path'].apply(lambda x: os.path.join('Ground truth', os.path.basename(x)))
        self.df_ms['path'] = self.df_ms['path'].apply(lambda x: os.path.join('Measurement', os.path.basename(x)))
        self.df_rc['path'] = self.df_rc['path'].apply(lambda x: os.path.join('Reconstruction', os.path.basename(x)))
        self.df_nl['path'] = self.df_nl['path'].apply(lambda x: os.path.join('No learning', os.path.basename(x)))

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot and optinally save the reconstructions.
        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        
        post_str = "Training" if train else "Eval"

        plot_images = self.plot_images and ((epoch + 1) % self.freq_plot == 0)
        save_images = self.save_folder_im is not None

        if plot_images or save_images:
            if self.compare_no_learning:
                x_nl = self.no_learning_inference(y, physics)
            else:
                x_nl = None
            
            imgs_list=[]
            img_meas_list=[]
            for i in range(x.shape[0]):
                imgs, titles, grid_image, caption = prepare_images(
                        x[i], y[i], x_net[i], x_nl[i], rescale_mode=self.rescale_mode
                    )
                imgs_list.append(imgs)

                if not 'Measurement' in caption:
                    img_meas = preprocess_img(y[i])
                    img_meas_list.append(img_meas)
        if plot_images:
            plot(
                imgs,
                titles=titles,
                show=False,
                return_fig=plot_images,
                rescale_mode=self.rescale_mode,
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch, step=epoch)

        if save_images:
            # save images
            for l, imgs in enumerate(imgs_list):
                for k, img in enumerate(imgs):
                    img=img.unsqueeze(0)
                    for i in range(img.size(0)):
                        if 'Ground' in titles[k]:
                            fold, name = os.path.split(self.df_gt.iloc[self.img_counter]['path'])
                        elif 'Recon' in titles[k]:
                            fold, name = os.path.split(self.df_rc.iloc[self.img_counter]['path'])
                        elif 'No learn' in titles[k]:
                            fold, name = os.path.split(self.df_nl.iloc[self.img_counter]['path'])
                        img_name = f"{self.save_folder_im}/{fold}/"
                        # make dir
                        Path(img_name).mkdir(parents=True, exist_ok=True)
                        save_image(img, img_name + f"{name}")
            
                    if not 'Measurement' in caption and k==0:
                        fold, name = os.path.split(self.df_ms.iloc[self.img_counter]['path'])
                        img_name_meas = f"{self.save_folder_im}/{fold}/"
                        Path(img_name_meas).mkdir(parents=True, exist_ok=True)
                        save_image(img_meas_list[l], img_name_meas + f"{name}")
                self.img_counter += 1
        self.df_gt.to_csv(f"{self.save_folder_im}/Ground truth.csv", index=True)
        self.df_rc.to_csv(f"{self.save_folder_im}/Reconstruction.csv", index=True)
        self.df_nl.to_csv(f"{self.save_folder_im}/No learning.csv", index=True)
        self.df_ms.to_csv(f"{self.save_folder_im}/Measurement.csv", index=True)



        if self.conv_metrics is not None:
            plot_curves(
                self.conv_metrics,
                save_dir=f"{self.save_folder_im}/convergence_metrics/",
                show=False,
            )
            self.conv_metrics = None

    def test(self, test_dataloader, save_path=None, compare_no_learning=True):
        r"""
        Test the model.

        :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] test_dataloader: Test data loader(s) should provide a
            a signal x or a tuple of (x, y) signal/measurement pairs.
        :param str save_path: Directory in which to save the trained model.
        :param bool compare_no_learning: If ``True``, the linear reconstruction is compared to the network reconstruction.
        :returns: The trained model.
        """
        self.compare_no_learning = compare_no_learning
        self.setup_train()

        self.save_folder_im = save_path
        aux = self.wandb_vis
        self.wandb_vis = False

        self.reset_metrics()

        if not isinstance(test_dataloader, list):
            test_dataloader = [test_dataloader]

        self.current_iterators = [iter(loader) for loader in test_dataloader]

        batches = min([len(loader) - loader.drop_last for loader in test_dataloader])

        self.model.eval()
        for i in (
            progress_bar := tqdm(
                range(batches),
                ncols=150,
                disable=(not self.verbose or not self.show_progress_bar),
            )
        ):
            progress_bar.set_description(f"Test")
            self.step(0, progress_bar, train=False, last_batch=True)

        self.wandb_vis = aux

        if self.verbose:
            print("Test results:")

        out = {}
        for k, l in enumerate(self.logs_metrics_eval):

            if compare_no_learning:
                name = self.metrics[k].__class__.__name__ + " no learning"
                out[name] = self.logs_metrics_linear[k].avg
                out[name + "_std"] = self.logs_metrics_linear[k].std
                if self.verbose:
                    print(
                        f"{name}: {self.logs_metrics_linear[k].avg:.3f} +- {self.logs_metrics_linear[k].std:.3f}"
                    )

            name = self.metrics[k].__class__.__name__
            out[name] = l.avg
            out[name + "_std"] = l.std
            if self.verbose:
                print(f"{name}: {l.avg:.3f} +- {l.std:.3f}")

        return out
    def no_learning_inference(self, y, physics):
        r"""
        Perform the no learning inference.

        By default it returns the (linear) pseudo-inverse reconstruction given the measurement.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :returns: Reconstructed image.
        """

        y = y.to(self.device)
        if self.no_learning_method == "A_adjoint" and hasattr(physics, "A_adjoint"):
            if isinstance(physics, torch.nn.DataParallel):
                x_nl = physics.module.A_adjoint(y)
            else:
                x_nl = physics.A_adjoint(y)
        elif self.no_learning_method == "A_dagger" and hasattr(physics, "A_dagger"):
            if isinstance(physics, torch.nn.DataParallel):
                x_nl = physics.module.A_dagger(y)
            else:
                x_nl = physics.A_dagger(y)
        elif self.no_learning_method == "prox_l2" and hasattr(physics, "prox_l2"):
            if isinstance(physics, torch.nn.DataParallel):
                x_nl = physics.module.prox_l2(y)
            else:
                x_nl = physics.prox_l2(y)
        elif self.no_learning_method == "y":
            x_nl = y

        elif self.no_learning_method == "interpolation":
            factor = physics.factor
            h, w = y.shape[-2:]
            img_size = [h*factor, w*factor]
            x_nl = F.interpolate(y, size=img_size, mode='bicubic', align_corners=False)
        else:
            raise ValueError(
                f"No learning reconstruction method {self.no_learning_method} not recognized"
            )

        return x_nl

def mytest(
    model,
    test_dataloader,
    physics,
    metrics=dinv.metric.PSNR(),
    online_measurements=False,
    physics_generator=None,
    device="cpu",
    plot_images=False,
    save_folder=None,
    plot_convergence_metrics=False,
    verbose=True,
    rescale_mode="clip",
    show_progress_bar=True,
    no_learning_method="A_dagger",
    df=None,
    **kwargs,
):
    trainer = myTrainer(
        model,
        physics=physics,
        train_dataloader=None,
        eval_dataloader=None,
        optimizer=None,
        metrics=metrics,
        online_measurements=online_measurements,
        physics_generator=physics_generator,
        device=device,
        plot_images=plot_images,
        plot_convergence_metrics=plot_convergence_metrics,
        verbose=verbose,
        rescale_mode=rescale_mode,
        no_learning_method=no_learning_method,
        show_progress_bar=show_progress_bar,
        df=df,
        **kwargs,
    )

    return trainer.test(test_dataloader, save_path=save_folder)