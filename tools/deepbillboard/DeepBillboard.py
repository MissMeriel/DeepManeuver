import kornia
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import os
# import matplotlib.pyplot as plt
# import matplotlib
# import skimage.io as sio
# # from onnx2pytorch import ConvertModel
# from pathlib import Path
# from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize
# import torch.nn.functional as F
from torchvision.utils import save_image


class DataSequence(data.Dataset):
    def __init__(self, img_array, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_arr = img_array
        self.transform = transform

    def __len__(self):
        return len(self.img_arr)

    def __getitem__(self, idx):
        sample = np.array(self.img_arr[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample

class DeepBillboard():

    def __init__(self, model, seqpath, direction):
        self.model = model
        self.model = model
        self.seqpath = seqpath
        self.direction = direction
        self.sample_dir = os.getcwd() + "/sampledir"
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

    def stripleftchars(self, s):
        for i in range(len(s)):
            if s[i].isnumeric():
                return s[i:]
        return -1

    def draw_arrows(self, img, angle1, angle2=None):
        import cv2

        img = (img.transpose((1, 2, 0)) * 255).round().astype(np.uint8).copy()

        pt1 = (int(img.shape[1] / 2), img.shape[0])
        pt2_angle1 = (
            int(img.shape[1] / 2 - img.shape[0] / 3 * np.sin(angle1)),
            int(img.shape[0] - img.shape[0] / 3 * np.cos(angle1)),
        )
        img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 3)
        if angle2 is not None:
            pt2_angle2 = (
                int(img.shape[1] / 2 - img.shape[0] / 3 * np.sin(angle2)),
                int(img.shape[0] - img.shape[0] / 3 * np.cos(angle2)),
            )
            img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 3)

        return img.astype(np.float32).transpose((2, 0, 1)) / 255

    def input_diversification(self, imgs, device):
        import random
        imgs_rs = torch.clone(imgs)
        for i in range(imgs.shape[0]):
            r1 = random.randint(0, imgs.shape[0]-1)
            r2 = random.uniform(0.5,1.5)
            scale_factor = torch.tensor([[r2]]).float().to(device)
            temp = kornia.geometry.transform.scale(imgs[r1][None], scale_factor)
            imgs_rs = torch.cat((imgs_rs,temp))
        return imgs_rs


    def perturb_images(self, img_arr: torch.Tensor, img_patches: np.ndarray, model: nn.Module,
                       bb_size=5, iterations=400, noise_level=25, device=torch.device("cuda"), input_divers=False):
        patch_coords = img_patches[:, 1:].reshape((-1, 4, 2))
        pert_shape = c, h, w = 3, bb_size, bb_size

        src_coords = np.tile(
            np.array(
                [
                    [
                        [0.0, 0.0],
                        [w - 1.0, 0.0],
                        [0.0, h - 1.0],
                        [w - 1.0, h - 1.0],
                    ]
                ]
            ),
            (len(patch_coords), 1, 1),
        )
        src_coords = torch.from_numpy(src_coords).float()
        patch_coords = torch.from_numpy(patch_coords).float()
        # build the transforms to and from image patches
        perspective_transforms = kornia.geometry.transform.get_perspective_transform(src_coords, patch_coords).to(device)

        model = model.to(device)
        dataset = DataSequence(img_arr)
        data_loader = data.DataLoader(dataset, batch_size=len(dataset))
        shape = next(iter(data_loader)).shape[2:]
        orig_shape = shape
        mask_patch = torch.ones(len(perspective_transforms), *pert_shape).float().to(device)
        perturbation = (torch.ones(1, *pert_shape)-0.5).float().to(device)

        num_iters = iterations
        for i in range(num_iters):
            perturbation = perturbation.detach()
            perturbation.requires_grad = True
            blurred_pert = perturbation

            imgs = next(iter(data_loader)).to(device)
            perturbation_warp = kornia.geometry.transform.warp_perspective(
                torch.vstack([blurred_pert for _ in range(len(perspective_transforms))]),
                perspective_transforms,
                dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )
            warp_masks = kornia.geometry.transform.warp_perspective(
                mask_patch, perspective_transforms, dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )

            imgs = imgs * (1 - warp_masks)
            imgs += perturbation_warp * warp_masks
            imgs = torch.clamp(imgs + torch.randn(*imgs.shape).to(device)/noise_level, 0, 1)

            y = model(imgs)
            if self.direction == "left": # left
                loss = (y - torch.full_like(y, -1)).mean()
                # loss = F.mse_loss(y, torch.full_like(y, -1))  
            else: # right
                loss = -(y - torch.full_like(y, 1)).mean()
                # loss = F.mse_loss(y, torch.full_like(y, 1))  

            print(
                f"[iteration {i:5d}/{num_iters}] loss={loss.item():2.5f} max(angle)={y.max().item():2.5f} min(angle)={y.min().item():2.5f} mean(angle)={y.mean().item():2.5f} median(angle)={torch.median(y).item():2.5f}"
            )
            loss.backward()

            perturbation = torch.clamp(
                perturbation - torch.sign(perturbation.grad) / 100, 0, 1
            )  # a variation of the fast gradient sign method

            model.zero_grad()
        y = model(imgs)
        y = y.cpu()
        y = y.detach().numpy().reshape((y.shape[0]))
        with torch.no_grad():
            mask_patch = mask_patch.cpu()
            perspective_transforms = perspective_transforms.cpu()
            perturbation = blurred_pert.detach().cpu()
            bb = np.round(perturbation[-1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            model = model.cpu()
            imgs = next(iter(data_loader))
            perturbation_warp = kornia.geometry.transform.warp_perspective(
                torch.vstack([perturbation for _ in range(len(perspective_transforms))]),
                perspective_transforms,
                dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )
            warp_masks = kornia.geometry.transform.warp_perspective(
                mask_patch, perspective_transforms, dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )

            perturbed_imgs = imgs * (1 - warp_masks)
            perturbed_imgs += perturbation_warp.cpu() * warp_masks.cpu()

            orig_angles = model(imgs)
            pert_angles = model(perturbed_imgs)
            MAE = (orig_angles - pert_angles).mean()

            return bb, y, MAE
