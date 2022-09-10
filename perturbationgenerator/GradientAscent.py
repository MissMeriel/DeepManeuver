import argparse
import kornia
import numpy as np
import skimage.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# from onnx2pytorch import ConvertModel
from pathlib import Path
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

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

class GradientAscent():

    def __init__(self, model, seqpath, direction, outdir=None):
        self.model = model
        self.model = model
        self.seqpath = seqpath
        self.direction = direction
        if outdir is None:
            self.sample_dir = os.getcwd() + "/sampledir"
        else:
            self.sample_dir = outdir
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)
        self.calls = 0

    def stripleftchars(self, s):
        for i in range(len(s)):
            if s[i].isnumeric():
                return s[i:]
        return -1

    # angle1 in blue, angle2 in green
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

    # approach for modified DAVE2
    # def perturb_images(self, img_arr, img_patches, network_name, model, device=torch.device("cuda")):
    def perturb_images(self, img_arr, img_patches, model, bb_size=5, iterations=400,
                           noise_level=25, device=torch.device("cuda")):
        self.calls = len(img_arr)
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
        perspective_transforms = kornia.get_perspective_transform(src_coords, patch_coords).to(device)
        patch_transforms = kornia.get_perspective_transform(patch_coords, src_coords)

        model = model.to(device)
        dataset = DataSequence(
            img_arr, #transform = Compose([ToPILImage(), ToTensor()])
        )
        data_loader = data.DataLoader(dataset, batch_size=len(dataset))
        shape = next(iter(data_loader)).shape[2:]
        orig_shape = shape
        mask_patch = torch.ones(len(perspective_transforms), *pert_shape).float().to(device)
        # initial sign is just the sign in the last image of
        # the sequence (because it is usually the biggest)
        # print("len(dataset):",len(dataset))
        # print("patch_transforms.shape:", patch_transforms.shape)
        # print("dsize:",pert_shape[1:])
        # perturbation = kornia.warp_perspective(
        #     next(
        #         iter(
        #             data.DataLoader(
        #                 DataSequence(
        #                     img_arr, transform=Compose([ToPILImage(), ToTensor()])
        #                 ),
        #                 batch_size=len(dataset),
        #             )
        #         )
        #     ),
        #     patch_transforms,
        #     dsize=pert_shape[1:],
        # )[-1:]
        perturbation = torch.ones(1, *pert_shape).float().to(device)
        # can try blurring it if you want
        # perturbation = torch.rand_like(mask_patch[0])[None]
        # perturbation = torch.ones_like(mask_patch[0])[None] / 2
        # perturbation = torch.zeros_like(mask_patch[0])[None]
        # perturbation = torch.ones_like(mask_patch[0])[None]
        # perturbation = kornia.filters.max_blur_pool2d(perturbation, 3, ceil_mode=True)
        # perturbation = kornia.filters.median_blur(perturbation, (3, 3))
        # perturbation = kornia.resize(perturbation, pert_shape[1:]).to(device)
        num_iters = iterations
        for i in range(num_iters):
            perturbation = perturbation.detach()
            perturbation.requires_grad = True
            blurred_pert = perturbation
            # can try blurring it to get smoother images (not so noisy)
            # gauss = kornia.filters.GaussianBlur2d((5, 5), (5.5, 5.5))
            # blurred_pert = gauss(perturbation)
            # blurred_pert = kornia.filters.blur_pool2d(perturbation, 3)
            # blurred_pert = kornia.filters.max_blur_pool2d(perturbation, 3, ceil_mode=True)
            # blurred_pert = kornia.filters.median_blur(perturbation, (3, 3))
            # blurred_pert = kornia.filters.median_blur(perturbation, (10, 10))
            # blurred_pert = kornia.filters.box_blur(perturbation, (3, 3))
            # blurred_pert = kornia.filters.box_blur(perturbation, (5, 5))
            # blurred_pert = kornia.resize(blurred_pert, perturbation.shape[2:])

            imgs = next(iter(data_loader)).to(device)
            y_orig = model(imgs)
            perturbation_warp = kornia.warp_perspective(
                torch.vstack([blurred_pert for _ in range(len(perspective_transforms))]),
                perspective_transforms,
                dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )
            warp_masks = kornia.warp_perspective(
                mask_patch, perspective_transforms, dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )
            # warp_masks = kornia.resize(warp_masks, shape)
            # perturbation_warp = kornia.resize(perturbation_warp, shape)
            imgs = imgs * (1 - warp_masks)
            imgs += perturbation_warp * warp_masks
            # save_image(imgs, self.sample_dir / f"pert_imgs_{i}.png")
            # save_image(perturbation, self.sample_dir / f"pert_{i}.png")
            #blurred_pert = kornia.filters.median_blur(perturbation, (3, 3))

            imgs = torch.clamp(imgs + torch.randn(*imgs.shape).to(device)/noise_level, 0, 1)
            y = model(imgs)
            if self.direction == "left":
                # loss = (y.flatten() - y_orig).mean()
                loss = (y.flatten() - -1 * torch.ones(*imgs.shape).to(device)).mean()

                # loss = F.mse_loss(y, torch.full_like(y, -1))  # left
            else:
            #     # loss = -(y.flatten() - y_orig).mean()
                loss = -(y.flatten() - torch.ones(*imgs.shape).to(device)).mean()
                # loss = F.mse_loss(y, torch.full_like(y, 1))  # right
            # loss = max(-(y - y_orig).mean(), (y - y_orig).mean())
            # loss = max(F.mse_loss(y, torch.full_like(y, -1)), F.mse_loss(y, torch.full_like(y, 1)), key=abs)

            print(
                f"[iteration {i:5d}/{num_iters}] loss={loss.item():2.5f} max(angle)={y.max().item():2.5f} min(angle)={y.min().item():2.5f} mean(angle)={y.mean().item():2.5f} median(angle)={torch.median(y).item():2.5f}"
            )
            loss.backward()

            # can try doing different things for the gradient descent step
            # perturbation = torch.clamp(
            #     perturbation - perturbation.grad, 0, 1
            # )  # simply follow the gradients
            # perturbation = torch.clamp(
            #     perturbation - perturbation.grad * 0.01, 0, 1
            # )  # try scaling the gradients
            perturbation = torch.clamp(
                perturbation - torch.sign(perturbation.grad) / 100, 0, 1
            )  # a variation of the fast gradient sign method
            # smoothed_gradients = kornia.filters.gaussian_blur2d(
            #     0.5 * perturbation.grad / torch.linalg.norm(perturbation.grad),
            #     (5, 5),
            #     (3, 3),
            # )  # smooth the gradients first to get less noisy billboards
            # perturbation = torch.clamp(perturbation - smoothed_gradients, 0, 1)
            model.zero_grad()
        y = model(imgs)
        y = y.cpu()
        y = y.detach().numpy().reshape((y.shape[0]))
        with torch.no_grad():
            mask_patch = mask_patch.cpu()
            perspective_transforms = perspective_transforms.cpu()
            # perturbation = perturbation.detach().cpu()
            # blurred_pert = perturbation.cpu()
            perturbation = blurred_pert.detach().cpu()

            # blurred_pert = kornia.filters.median_blur(perturbation, (3, 3))
            # blurred_pert = kornia.resize(blurred_pert, perturbation.shape[2:])
            # bb = np.round(blurred_pert[-1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # print(perturbation.shape)
            # perturbation = kornia.resize(perturbation, (100,100))
            bb = np.round(perturbation[-1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # # bb = np.array(perturbation.permute(2, 3, 1, 0).tolist())* 255
            # # bb = bb[:,:,:,0].astype(np.uint8)
            # bb = np.flipud(bb)
            # plt.title("perturbation")
            # plt.imshow(bb)
            # plt.show()
            # plt.pause(0.01)

            imgs = next(iter(data_loader))
            perturbation_warp = kornia.warp_perspective(
                torch.vstack([perturbation for _ in range(len(perspective_transforms))]),
                perspective_transforms,
                dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )
            warp_masks = kornia.warp_perspective(
                mask_patch, perspective_transforms, dsize=orig_shape,
                mode="nearest",
                align_corners=True
            )
            # warp_masks = kornia.resize(warp_masks, shape)
            # perturbation_warp = kornia.resize(perturbation_warp, shape)
            perturbed_imgs = imgs * (1 - warp_masks)
            perturbed_imgs += perturbation_warp * warp_masks

            model = model.cpu()
            orig_angles = model(imgs)
            pert_angles = model(perturbed_imgs)
            print("Original outputs:", orig_angles)
            print("Adversarial outputs:", pert_angles)
            arrowed_imgs = []
            orig_arrowed_imgs = []
            for img, pert_img, a1, a2 in zip(
                imgs, perturbed_imgs, orig_angles, pert_angles
            ):
                img_processed = self.draw_arrows(img.numpy(), a1.numpy())
                orig_arrowed_imgs.append(img_processed)
                img_processed = self.draw_arrows(pert_img.numpy(), a1.numpy(), a2.numpy())
                arrowed_imgs.append(img_processed)
            save_image(
                torch.from_numpy(np.asarray(arrowed_imgs)), self.sample_dir +"/"+ f"arrows_{self.calls}.png"
            )
            # save_image(
            #     torch.from_numpy(np.asarray(orig_arrowed_imgs)),
            #     self.sample_dir +"/"+ f"arrows_orig_{self.calls}.png",
            # )
            # filename = f"{self.sample_dir}/inputs.txt"
            # with open(filename, "w") as f:
            #     f.write("TIMESTEP,ORIG_INPUT,PERT_INPUT\n")
            #     for i,oa,pa in zip(range(len(orig_angles)), orig_angles, pert_angles):
            #         f.write(f"{i},{oa},{pa}\n")
            model = model.to(device)
            return bb, y, perturbed_imgs, arrowed_imgs
