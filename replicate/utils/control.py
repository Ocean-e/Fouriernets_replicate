# functions to control training and testing using built networls

import datetime
import time
import logging
import os
import torch
import torch.cuda.comm
import torch.nn as nn
import torchvision.transforms.functional as F

import lpips as lp
from .output_control import *

def train_rgb_recon(
    deconv,
    opt, 
    dataloader,
    devices,
    mses,
    num_iterations,
    lr_scheduler=None,
    lpips_weight=0,
    lpips_step_milestones=[],
    lpips_step_size=0.1,
    lpips_losses=[],
    losses=[],
    checkpoint_interval=10,
    snapshot_interval=10,
    validate_mses=[],
    validate_args=None,
    it=0,
):
    """
    Trains a reconstruction network for DLMD.
    """
    # define constants
    scheduler = lr_scheduler is not None

    # define logging messages
    log_string = "[{}] iter: {}, loss: {}"
    profile_string = "[{}] {}"

    # create mse loss
    mse = nn.MSELoss()

    # create perceptual loss
    perceptual = lp.LPIPS().to(devices[0])

    # initialize end time
    end_time = None

    # train loop
    while it < num_iterations:
        for batch in dataloader: 
            start_time = time.perf_counter()
            diffused_images = batch["image"].to(devices[0])
            ground_truth_images = batch["label"].to(devices[0])
            if end_time is None:
                logging.info(
                    profile_string.format(start_time, "done loading new sample")
                )
            else:
                logging.info(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            # clear gradients
            opt.zero_grad()
            # reconstruct images
            predicted_images = deconv(diffused_images)
            recon_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (recon_time - start_time),
                    "done reconstructing sample",
                )
            )
            # calculate loss
            mse_loss = mse(predicted_images, ground_truth_images)
            if lpips_weight > 0:
                lpips_loss = perceptual(predicted_images, ground_truth_images)
                lpips_loss=torch.mean(lpips_loss)
                if lpips_loss<=0:
                    print('ERROR:lpips<=0')
                loss = ((1.0 - lpips_weight) * mse_loss) + (lpips_weight * lpips_loss)
            else:
                loss = mse_loss
            loss_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (loss_time - recon_time),
                    "done calculating loss",
                )
            )
            # backpropagate gradients
            loss.backward()
            backward_time = time.perf_counter()
            logging.info(
                profile_string.format((backward_time - loss_time), "done with backward")
            )
            # update weights
            opt.step()
            if scheduler:
                # advance learning rate schedule
                lr_scheduler.step()
            opt_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (opt_time - backward_time), "done with optimizer step"
                )
            )
            # checkpoint/snapshot
            mses.append(mse_loss.detach().cpu().item())
            if lpips_weight > 0:
                lpips_losses.append(lpips_loss.detach().cpu().item())
            logging.info(
                log_string.format(
                    datetime.datetime.now(), it, loss.detach().cpu().item()
                )
            )
            losses.append(loss.detach().cpu().item())
            # checkpoint:
            if it % 10000 == 0 and it != 0 and validate_args is not None: 
                per_sample_val_losses = test_rgb_recon(deconv, **validate_args)
                validate_mses.append(per_sample_val_losses)
            if it % checkpoint_interval == 0 and it != 0:
                logging.info("checkpointing...")
                checkpoint(
                    deconv,
                    opt,
                    {
                        "mses": mses,
                        "lpips_losses": lpips_losses,
                        "losses":losses,
                        "validate_mses": validate_mses,
                    },
                    it,
                    module_name="deconv",
                )
            if (it % snapshot_interval == 0) or (it == num_iterations - 1):
                logging.info("snapshotting...")
                snapshot(
                    deconv,
                    opt,
                    {
                        "mses": mses,
                        "lpips_losses": lpips_losses,
                        "losses":losses,
                        "validate_mses": validate_mses,
                    },
                    it,
                    ground_truth_images.detach().cpu(),
                    predicted_images.detach().cpu(),
                    diffused_images.detach().cpu(),
                    module_name="deconv",
                )

            # update iteration count and check for end
            it += 1
            if (len(lpips_step_milestones) > 0) and (it == lpips_step_milestones[0]):
                _ = lpips_step_milestones.pop(0)
                lpips_weight += lpips_step_size
                lpips_weight = min(1.0, lpips_weight)
            end_time = time.perf_counter()
            logging.info(
                profile_string.format((end_time - start_time), "done with loop")
            )
            if it >= num_iterations:
                break


def test_rgb_recon(
    deconv,
    dataloader,
    devices,
    save_dir="./",
):
    """
    Trains a reconstruction network for DLMD.
    """
    
    # set model to eval mode
    deconv.eval()
    # define logging messages
    log_string = "[{}] sample: {}, mse_loss: {}"
    profile_string = "[{}] {}"
    log_fname = os.path.join(save_dir, "out.log")

    # create mse loss
    mse = nn.MSELoss()

    # initialize losses
    mses = []
    lpipses=[]
    ssims=[]
    psnrs=[]

    # initialize counter
    counter = 0

    # initialize log file
    f = open(log_fname, "w")

    # initialize end time
    end_time = None

    # train loop
    with torch.no_grad():
        for single in dataloader:
            # WARN(dip): assuming batch size for test dataloader is 1
            start_time = time.perf_counter()
            diffused_image = single["image"].to(devices[0])
            ground_truth_image = single["label"].to(devices[0])
            if end_time is None:
                f.write(profile_string.format(start_time, "done loading new sample"))
            else:
                f.write(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            # reconstruct images
            predicted_image = deconv(diffused_image)
            recon_time = time.perf_counter()
            f.write(
                profile_string.format(
                    (recon_time - start_time),
                    "done reconstructing sample",
                )
            ) 

            perceptual = lp.LPIPS(net="alex").to(devices[0])
            
            # calculate loss
            #ssim = F.ssim(predicted_image, ground_truth_image, data_range=1.0, size_average=True)
            #psnr = F.psnr(predicted_image, ground_truth_image, data_range=1.0)
            mse_loss = mse(predicted_image, ground_truth_image)
            lpips_loss = perceptual(predicted_image, ground_truth_image)

            loss_time = time.perf_counter()
            f.write(
                profile_string.format(
                    (loss_time - recon_time),
                    "done calculating loss",
                )
            )
            # log results
            # TODO(dip): call these by MSE/more specific names
            mses.append(mse_loss.detach().cpu().item())
            lpipses.append(lpips_loss.detach().cpu().item())
            # ssims.append(ssim.detach().cpu().item())
            # psnrs.append(psnr.detach().cpu().item())
            # save results
            f.write(log_string.format(datetime.datetime.now(), counter, mses[-1]))
            torch.save(predicted_image.detach().cpu(), f"{save_dir}/recon{counter}.pt")
            torch.save(ground_truth_image.detach().cpu(), f"{save_dir}/sam{counter}.pt")
            torch.save(diffused_image.detach().cpu(), f"{save_dir}/im{counter}.pt")
            # update iteration count
            counter += 1
            end_time = time.perf_counter()
            f.write(
                profile_string.format((end_time - start_time), "done testing sample")
            )
        mse=sum(mses)/len(mses)
        lpips=sum(lpipses)/len(lpipses)
        # ssim=sum(ssims)/len(ssims)
        # psnr=sum(psnrs)/len(psnrs)

        print(f'mse={mse}\n lpips={lpips}\n')

    f.close()
    # return model to train mode
    deconv.train()