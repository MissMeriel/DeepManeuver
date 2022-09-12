import numpy as np
import os, random, copy
import kornia
import torch
import torch.utils.data as data
import torch.nn.functional as F
# from pathlib import Path
# from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize
from torchvision.utils import save_image
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import skimage.io as sio
# import argparse

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

class DeepManeuver():

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
            angle2 = -angle2
            pt2_angle2 = (
                int(img.shape[1] / 2 - img.shape[0] / 3 * np.sin(angle2)),
                int(img.shape[0] - img.shape[0] / 3 * np.cos(angle2)),
            )
            img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 3)
        return img.astype(np.float32).transpose((2, 0, 1)) / 255

    def input_diversification(self, imgs, y_orig, device):
        imgs_rs = imgs
        y_orig_exp = copy.deepcopy(y_orig)
        for i in range(int(imgs.shape[0] / 2)):
            r1 = random.randint(0, imgs.shape[0]-1)
            r2 = random.uniform(0.5, 1.5)
            scale_factor = torch.tensor([[r2]]).float().to(device)
            temp = kornia.geometry.transform.scale(imgs[r1][None], scale_factor)
            imgs_rs = torch.cat((imgs_rs,temp))
            y_orig_exp = torch.cat((y_orig_exp, y_orig[r1][None]),dim=0)
        return imgs_rs, y_orig_exp


    def perturb_images(self, img_arr, img_patches, model, steering_vector,
                       bb_size=5, iterations=400, noise_level=25,
                       device=torch.device("cuda"), last_billboard=None, loss_fxn="MDirE", input_divers=False):
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
        perspective_transforms = kornia.geometry.transform.get_perspective_transform(src_coords, patch_coords).to(device)
        patch_transforms = kornia.geometry.transform.get_perspective_transform(patch_coords, src_coords)

        model = model.to(device)
        steering_vector = steering_vector.to(device)
        dataset = DataSequence(
            img_arr, #transform = Compose([ToPILImage(), ToTensor()])
        )
        data_loader = data.DataLoader(dataset, batch_size=len(dataset))
        shape = next(iter(data_loader)).shape[2:]
        orig_shape = shape
        mask_patch = torch.ones(len(perspective_transforms), *pert_shape).float().to(device)

        perturbation = (torch.ones(1, *pert_shape)-0.5).float().to(device)

        # line_mask = torch.ones(1, *pert_shape).float().to(device)
        # for i in range(line_mask.shape[2]):
        #     if i % 2:
        #         line_mask[:, :, i, :] = line_mask[:, :, i, :] * 0
        steering_vector_orig = copy.deepcopy(steering_vector)
        for i in range(iterations):
            # if input_divers:
            #     perturbation = perturbation * line_mask
            perturbation = perturbation.detach()
            perturbation.requires_grad = True

            imgs = next(iter(data_loader)).to(device)
            y_orig = model(imgs)
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
            imgs = imgs * (1 - warp_masks)
            imgs += perturbation_warp * warp_masks

            # if input_divers:
            #     imgs, steering_vector = self.input_diversification(imgs, steering_vector_orig, device)

            def extra_noise(imgs, noise_level, steering_vector_orig):
                imgs1 = torch.clamp(imgs + torch.randn(*imgs.shape).to(device) / noise_level, 0, 1)
                imgs2 = torch.clamp(imgs + torch.randn(*imgs.shape).to(device) / noise_level, 0, 1)
                imgs = torch.cat([imgs, imgs1])
                steering_vector = torch.cat([steering_vector_orig, steering_vector_orig])
                return imgs, steering_vector

            def interleave_noise(imgs, noise_level, sv):
                interleaved_vector = torch.empty((sv.shape[0]*2))
                interleaved_imgs = torch.empty((imgs.shape[0]*2, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
                for i in range(imgs.shape[0]*2, 2):
                        interleaved_imgs[i] = torch.clamp(imgs[i] + torch.randn(*imgs[i].shape).to(device) / noise_level,0, 1)
                        interleaved_imgs[i + 1] = torch.clamp(imgs[i + 1] + torch.randn(*imgs[i].shape).to(device) / noise_level, 0, 1)
                        interleaved_imgs[i + 2] = imgs[i]
                        interleaved_imgs[i + 3] = imgs[i + 1]
                        interleaved_vector[i] = sv[i]
                        interleaved_vector[i + 1] = sv[i + 1]
                        interleaved_vector[i + 2] = sv[i]
                        interleaved_vector[i + 3] = sv[i + 1]
                print(f"{len(interleaved_imgs)=}, {len(interleaved_vector)=}")
                return interleaved_imgs.float().to(device), interleaved_vector.float().to(device)

            imgs = torch.clamp(imgs + torch.randn(*imgs.shape).to(device) / noise_level, 0, 1)
            # imgs, steering_vector = extra_noise(imgs, noise_level, steering_vector_orig)

            y = model(imgs)
            assert len(steering_vector.shape)
            assert y.shape[0] == steering_vector.shape[0]
            assert y.shape[1] == 1
            if self.direction == "left" and loss_fxn == "MDirE":
                loss = (y.flatten() - steering_vector).mean() # usual loss fxn
            elif self.direction == "right" and loss_fxn == "MDirE":
                loss = -(y.flatten() - steering_vector).mean()
            elif loss_fxn == "MSE":
                loss = F.mse_loss(y.flatten(), steering_vector)
                # loss = F.mse_loss(y, torch.full_like(y, -1))  # left
                # loss = F.mse_loss(y, torch.full_like(y, 1))  # right
            elif loss_fxn == "MAbsE":
                loss = abs(y.flatten() - steering_vector).mean()
                # loss = -F.smooth_l1_loss(y.flatten(), steering_vector, beta=0.1)
                # loss = F.binary_cross_entropy(F.sigmoid(y.flatten()), steering_vector)
            elif loss_fxn == "inv23" and self.direction == "left":
                # decay_factors = 1 / torch.sqrt(torch.Tensor([i for i in range(1,y.flatten().shape[0]+1)])).to(device)
                decay_factors = 1 / torch.pow(torch.Tensor([i for i in range(1, y.flatten().shape[0] + 1)]), 1/100).to(device)
                loss = (decay_factors * (y.flatten() - steering_vector)).mean()
            elif loss_fxn == "inv23" and self.direction == "right":
                # decay_factors = 1 / torch.sqrt(torch.Tensor([i for i in range(1,y.flatten().shape[0]+1)])).to(device)
                decay_factors = 1 / torch.pow(torch.Tensor([i for i in range(1, y.flatten().shape[0] + 1)]), 1/100).to(device)
                loss = -(decay_factors * (y.flatten() - steering_vector)).mean()

            print(
                f"[iteration {i:5d}/{iterations}] loss={loss.item():2.5f} max(angle)={y.max().item():2.5f} min(angle)={y.min().item():2.5f} mean(angle)={y.mean().item():2.5f} median(angle)={torch.median(y).item():2.5f}"
            )
            loss.backward()

            # can try doing different things for the gradient descent step
            # perturbation = torch.clamp(
            #     perturbation - perturbation.grad, 0, 1
            # )  # simply follow the gradients
            # perturbation = torch.clamp(
            #     perturbation - perturbation.grad / torch.linalg.norm(perturbation.grad), 0, 1
            # )  # try scaling the gradients
            perturbation = torch.clamp(
                perturbation - torch.sign(perturbation.grad) / 100, 0, 1
            )  # a variation of the fast gradient sign method
            # smoothed_gradients = kornia.filters.gaussian_blur2d(
            #     0.5 * perturbation.grad / torch.linalg.norm(perturbation.grad),
            #     (5, 5),
            #     (3, 3),
            # )  # smooth the gradients first to get less noisy billboards
            # NOTE: only use smoothed gradients on large billboards, otherwise billboard will all be same color due to kernel size
            # perturbation = torch.clamp(perturbation - smoothed_gradients, 0, 1)
            model.zero_grad()
        print(
            f"[iteration {i:5d}/{iterations}] loss={loss.item():2.5f} max(angle)={y.max().item():2.5f} min(angle)={y.min().item():2.5f} mean(angle)={y.mean().item():2.5f} median(angle)={torch.median(y).item():2.5f}"
        )
        y = model(imgs)
        y = y.cpu()
        y = y.detach().numpy().reshape((y.shape[0]))
        steering_vector = steering_vector.cpu()

        with torch.no_grad():
            mask_patch = mask_patch.cpu()
            perspective_transforms = perspective_transforms.cpu()
            perturbation = perturbation.detach().cpu()
            bb = np.round(perturbation[-1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

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
            perturbed_imgs += perturbation_warp * warp_masks

            model = model.cpu()
            imgs = imgs.cpu()
            orig_angles = model(imgs)
            pert_angles = model(perturbed_imgs)
            MAE = (orig_angles - pert_angles).mean()

            # arrowed_imgs = []
            # orig_arrowed_imgs = []
            # for img, pert_img, a1, a2 in zip(
            #     imgs, perturbed_imgs, orig_angles, pert_angles
            # ):
            #     img_processed = self.draw_arrows(img.numpy(), a1.numpy())
            #     orig_arrowed_imgs.append(img_processed)
            #     img_processed = self.draw_arrows(pert_img.numpy(), a1.numpy(), a2.numpy())
            #     arrowed_imgs.append(img_processed)
            # save_image(
            #     torch.from_numpy(np.asarray(arrowed_imgs)), self.sample_dir +"/"+ f"arrows_{calls}.png"
            # )
            # save_image(
            #     torch.from_numpy(np.asarray(orig_arrowed_imgs)),
            #     self.sample_dir +"/"+ "arrows_orig.png",
            # )

            return bb, y, MAE
