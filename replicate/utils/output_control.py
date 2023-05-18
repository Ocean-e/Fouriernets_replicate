import torch
import torch.cuda.comm


def checkpoint(
    module_container,
    opt,
    loss_dict,
    it,
    fname="latest.pt",
    module_name="micdeconv",
):
    """
    Saves the current model/optimizer state/iteration number/loss to fname.
    """
    checkpoint_dict = {
        f"{module_name}_state_dict": module_container.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "it": it,
    }
    checkpoint_dict.update(loss_dict)  
    torch.save(checkpoint_dict, fname)


def snapshot(
    module_container,
    opt,
    loss_dict,
    it,
    sam,
    recon,
    im,
    psf=None,
    phase_mask_angle=None,
    mirror_phase=None,
    save_dir="snapshots",
    module_name="micdeconv",
):
    """
    Saves the current model/optimizer state/iteration number/loss to fname. 
    Save the current sample, reconstruction, and image.
    Optionally saves the current PSF, phase mask,and/or mirror phase.
    """
    checkpoint(
        module_container,
        opt,
        loss_dict,
        it,
        fname=f"{save_dir}/state{it}.pt",
        module_name=module_name,
    )
    torch.save(sam.squeeze(), f"{save_dir}/sam{it}.pt") 
    torch.save(recon.squeeze(), f"{save_dir}/recon{it}.pt")
    if im is not None:
        torch.save(im.squeeze(), f"{save_dir}/im{it}.pt")
    if psf is not None:
        torch.save(psf.detach().cpu(), f"{save_dir}/psf{it}.pt")
    if phase_mask_angle is not None:
        torch.save(phase_mask_angle.detach().cpu(), f"{save_dir}/phase_mask{it}.pt")
    if mirror_phase is not None:
        torch.save(mirror_phase.detach().cpu(), f"{save_dir}/mirror_phase{it}.pt")
    torch.cuda.empty_cache()