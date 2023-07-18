import argparse
import kornia
import numpy as np
# import onnx
import skimage.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# from onnx2pytorch import ConvertModel
from pathlib import Path
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda
from torchvision.utils import save_image
from DAVE2 import DAVE2Model
from DAVE2pytorch import DAVE2PytorchModel
# import tensorflow as tf
# import mmdnn
# import tf2onnx.convert
# import keras2onnx
# import netron

def stripleftchars(s):
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1


class DataSequence(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform

        image_paths = []
        for p in Path(root).iterdir():
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                image_paths.append(p)
        image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        sample = sio.imread(img_name)

        if self.transform:
            sample = self.transform(sample)

        return sample


def draw_arrows(img, angle1, angle2=None):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=Path)
    parser.add_argument("seqpath", type=Path)
    parser.add_argument(
        "--direction", type=str, choices=["left", "right"], default="right"
    )
    return parser.parse_args()

def main(args):
    print((args))
    sample_dir = Path("samples") / args.seqpath.stem
    sample_dir.mkdir(exist_ok=True, parents=True)

    # read in coordinates of image patches corresponding to the modifiable sign
    patch_coords = np.loadtxt(args.seqpath / "coordinates.txt")[:, 1:].reshape(
        (-1, 4, 2)
    )
    h_ = (patch_coords[:, :, 1].max(1) - patch_coords[:, :, 1].min(1)).max()
    w_ = (patch_coords[:, :, 0].max(1) - patch_coords[:, :, 0].min(1)).max()

    # can play around with different sizes for the patch
    # pert_shape = c, h, w = 3, round(h_) // 3, round(w_) // 3
    if "Kitti" in args.seqpath.stem:
        orig_h, orig_w = 512, 1392
    elif "Dave" in args.seqpath.stem:
        orig_h, orig_w = 256, 455
    elif "Udacity" in args.seqpath.stem:
        orig_h, orig_w = 480, 640
    else:
        raise RuntimeError("Unknown dataset")
    pert_shape = c, h, w = 3, round(100 * h_ / orig_h), round(100 * w_ / orig_w)
    # pert_shape = c, h, w = 3, round(h_), round(w_)
    print((orig_h, orig_w), pert_shape)
    # pert_shape = c, h, w = 3, 32, 32

    # we're going to map from a CxHxW image to the patch in each image
    src_coords = np.tile(
        np.array(
            [
                [
                    [0.0, h - 1.0],
                    [w - 1.0, h - 1.0],
                    [0.0, 0.0],
                    [w - 1.0, 0.0],
                ]
            ]
        ),
        (len(patch_coords), 1, 1),
    )
    print(patch_coords.shape, src_coords.shape, pert_shape)

    src_coords = torch.from_numpy(src_coords).float()
    patch_coords = torch.from_numpy(patch_coords).float()
    # build the transforms to and from image patches
    perspective_transforms = kornia.get_perspective_transform(src_coords, patch_coords)
    patch_transforms = kornia.get_perspective_transform(patch_coords, src_coords)

    onnx_model = onnx.load(args.modelpath)
    # I haven't used this onnx2pytorch package before,
    # so hopefully it works
    model = ConvertModel(onnx_model, experimental=True)

    # data needs to be pre-processed
    # easier to just add it to the front of the network in this case
    class Preprocess(nn.Module):
        def forward(self, x):
            x = x * 255.0
            x[:, 0] -= 103.939
            x[:, 1] -= 116.779
            x[:, 2] -= 123.68
            return x

    # DAVE network should have an 2*Atan operation on the end
    # that is not included in our ONNX model, so add it in here
    class Atan(nn.Module):
        def forward(self, x):
            return 2 * torch.atan(x)

    # construct the full model with pre-processing built in
    model = nn.Sequential(Preprocess(), model, Atan())
    print(model)

    dataset = DataSequence(
        args.seqpath, transform=Compose([ToPILImage(), Resize((100, 100)), ToTensor()])
    )
    data_loader = data.DataLoader(dataset, batch_size=len(dataset))
    shape = next(iter(data_loader)).shape[2:]
    orig_shape = sio.imread(dataset.image_paths[0]).shape[:2]
    print(shape, orig_shape)
    mask_patch = torch.ones(len(perspective_transforms), *pert_shape).float()
    # initial sign is just the sign in the last image of
    # the sequence (because it is usually the biggest)
    perturbation = kornia.warp_perspective(
        next(
            iter(
                data.DataLoader(
                    DataSequence(
                        args.seqpath, transform=Compose([ToPILImage(), ToTensor()])
                    ),
                    batch_size=len(dataset),
                )
            )
        ),
        patch_transforms,
        dsize=pert_shape[1:],
    )[-1:]
    # can try blurring it if you want
    # perturbation = torch.rand_like(mask_patch[0])[None]
    # perturbation = torch.ones_like(mask_patch[0])[None] / 2
    # perturbation = torch.zeros_like(mask_patch[0])[None]
    # perturbation = torch.ones_like(mask_patch[0])[None]
    # perturbation = kornia.filters.max_blur_pool2d(perturbation, 3, ceil_mode=True)
    # perturbation = kornia.filters.median_blur(perturbation, (3, 3))
    perturbation = kornia.resize(perturbation, pert_shape[1:])
    num_iters = 200
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
        blurred_pert = kornia.resize(blurred_pert, perturbation.shape[2:])

        imgs = next(iter(data_loader))
        perturbation_warp = kornia.warp_perspective(
            torch.vstack([blurred_pert for _ in range(len(perspective_transforms))]),
            perspective_transforms,
            dsize=orig_shape,
        )
        warp_masks = kornia.warp_perspective(
            mask_patch, perspective_transforms, dsize=orig_shape
        )
        warp_masks = kornia.resize(warp_masks, shape)
        perturbation_warp = kornia.resize(perturbation_warp, shape)
        imgs = imgs * (1 - warp_masks)
        imgs += perturbation_warp * warp_masks
        save_image(imgs, sample_dir / f"pert_imgs_{i}.png")
        save_image(perturbation, sample_dir / f"pert_{i}.png")

        y = model(imgs)
        if args.direction == "left":
            loss = F.mse_loss(y, torch.full_like(y, 1.5))  # left
        else:
            loss = F.mse_loss(y, torch.full_like(y, -1.5))  # right
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
        # perturbation = torch.clamp(
        #     perturbation - np.sign(perturbation.grad) / 100, 0, 1
        # )  # a variation of the fast gradient sign method
        smoothed_gradients = kornia.filters.gaussian_blur2d(
            0.5 * perturbation.grad / torch.linalg.norm(perturbation.grad),
            (5, 5),
            (3, 3),
        )  # smooth the gradients first to get less noisy billboards
        perturbation = torch.clamp(perturbation - smoothed_gradients, 0, 1)
        model.zero_grad()

    with torch.no_grad():
        perturbation = perturbation.detach()
        blurred_pert = perturbation
        blurred_pert = kornia.filters.median_blur(perturbation, (3, 3))
        blurred_pert = kornia.resize(blurred_pert, perturbation.shape[2:])
        imgs = next(iter(data_loader))
        perturbation_warp = kornia.warp_perspective(
            torch.vstack([blurred_pert for _ in range(len(perspective_transforms))]),
            perspective_transforms,
            dsize=orig_shape,
        )
        warp_masks = kornia.warp_perspective(
            mask_patch, perspective_transforms, dsize=orig_shape
        )
        warp_masks = kornia.resize(warp_masks, shape)
        perturbation_warp = kornia.resize(perturbation_warp, shape)
        perturbed_imgs = imgs * (1 - warp_masks)
        perturbed_imgs += perturbation_warp * warp_masks

        orig_angles = model(imgs)
        pert_angles = model(perturbed_imgs)
        print("Original outputs:", orig_angles)
        print("Adversarial outputs:", pert_angles)
        arrowed_imgs = []
        orig_arrowed_imgs = []
        for img, pert_img, a1, a2 in zip(
            imgs, perturbed_imgs, orig_angles, pert_angles
        ):
            img_processed = draw_arrows(img.numpy(), a1.numpy())
            orig_arrowed_imgs.append(img_processed)
            img_processed = draw_arrows(pert_img.numpy(), a1.numpy(), a2.numpy())
            arrowed_imgs.append(img_processed)
        save_image(
            torch.from_numpy(np.asarray(arrowed_imgs)), sample_dir / "arrows.png"
        )
        save_image(
            torch.from_numpy(np.asarray(orig_arrowed_imgs)),
            sample_dir / "arrows_orig.png",
        )

# approach for modified DAVE2
def main2(args):
    # with open('H:/GitHub/DAVE2-Keras/BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-3.json') as f:
    #     model_json = f.read()
    # model = tf.keras.models.model_from_json(model_json)
    # exit(0)
    # onnx.save(onnx_model, 'DAVE2convertedmodel.onnx')
    # sm = DAVE2Model()
    # dual_model = sm.define_dual_model_BeamNG()
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-3-model.h5")
    # # print(dual_model.summary())
    # dualmodel_onnx = keras2onnx.convert_keras(dual_model, name=None, doc_string='', target_opset=None, channel_first_inputs=None)

    sample_dir = Path("samples") / args.seqpath.stem
    sample_dir.mkdir(exist_ok=True, parents=True)

    # read in coordinates of image patches corresponding to the modifiable sign
    patch_coords = np.loadtxt(args.seqpath / "coordinates.txt")
    patch_coords = patch_coords[:, 1:]
    patch_coords = patch_coords.reshape((-1, 4, 2))
    print("np.loadtxt(args.seqpath/coordinates.txt)", np.loadtxt(args.seqpath / "coordinates.txt").shape)
    print("patch_coords.shape:", patch_coords.shape)
    h_ = (patch_coords[:, :, 1].max(1) - patch_coords[:, :, 1].min(1)).max()
    w_ = (patch_coords[:, :, 0].max(1) - patch_coords[:, :, 0].min(1)).max()

    # can play around with different sizes for the patch
    # pert_shape = c, h, w = 3, round(h_) // 3, round(w_) // 3
    if "Kitti" in args.seqpath.stem:
        orig_h, orig_w = 512, 1392
    elif "Dave" in args.seqpath.stem:
        orig_h, orig_w = 256, 455
    elif "DaveMod" in args.seqpath.stem:
        orig_h, orig_w = 150, 200
    elif "Udacity" in args.seqpath.stem:
        orig_h, orig_w = 480, 640
    else:
        raise RuntimeError("Unknown dataset")
    pert_shape = c, h, w = 3, round(100 * h_ / orig_h), round(100 * w_ / orig_w)
    # pert_shape = c, h, w = 3, round(h_), round(w_)
    print((orig_h, orig_w), pert_shape)
    # pert_shape = c, h, w = 3, 32, 32

    # we're going to map from a CxHxW image to the patch in each image
    src_coords = np.tile(
        np.array(
            [
                [
                    [0.0, h - 1.0],
                    [w - 1.0, h - 1.0],
                    [0.0, 0.0],
                    [w - 1.0, 0.0],
                ]
            ]
        ),
        (len(patch_coords), 1, 1),
    )
    print(patch_coords.shape, src_coords.shape, pert_shape)

    src_coords = torch.from_numpy(src_coords).float()
    patch_coords = torch.from_numpy(patch_coords).float()
    print("src_coords.shape:", src_coords.shape)
    print("patch_coords.shape:", patch_coords.shape)
    # build the transforms to and from image patches
    perspective_transforms = kornia.get_perspective_transform(src_coords, patch_coords)
    patch_transforms = kornia.get_perspective_transform(patch_coords, src_coords)


    # I haven't used this onnx2pytorch package before,
    # so hopefully it works
    # onnx_model = onnx.load(args.modelpath)
    # sm = DAVE2Model()
    # model = sm.define_dual_model_BeamNG()
    # model.load_weights('H:/GitHub/DAVE2-Keras/BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-3-model.h5')
    # onnx_model, _ = tf2onnx.convert.from_keras(model)
    # onnx.save(onnx_model, 'davemod.onnx')
    # print("successfully converted using tf2onnx!")
    # netron.start('davemod.onnx')
    # model = ConvertModel(onnx_model, experimental=True)

    model = DAVE2PytorchModel().load("../dave_v1.pt")
    # convert DAVE2 pytorch to modified DAVE2
    # sm = DAVE2Model()
    # dual_model = sm.define_dual_model_BeamNG()
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-3-model.h5")
    # print(dual_model.summary())
    # dualmodel_onnx = keras2onnx.convert_keras(dual_model, name=None, doc_string='', target_opset=None, channel_first_inputs=None)

    # # data needs to be pre-processed
    # # easier to just add it to the front of the network in this case
    # class Preprocess(nn.Module):
    #     def forward(self, x):
    #         x = x * 255.0
    #         x[:, 0] -= 103.939
    #         x[:, 1] -= 116.779
    #         x[:, 2] -= 123.68
    #         return x
    #
    # # DAVE network should have an 2*Atan operation on the end
    # # that is not included in our ONNX model, so add it in here
    # class Atan(nn.Module):
    #     def forward(self, x):
    #         return 2 * torch.atan(x)

    # construct the full model with pre-processing built in
    # model = nn.Sequential(Preprocess(), model, Atan())
    print(model)

    dataset = DataSequence(
        # args.seqpath, transform=Compose([ToPILImage(), Resize((100, 100)), ToTensor()])
        args.seqpath, transform=Compose([ToPILImage(), Resize((150, 200)), ToTensor(), Lambda(lambda x: x/127.5-1.0)])
    )
    data_loader = data.DataLoader(dataset, batch_size=len(dataset))
    shape = next(iter(data_loader)).shape[2:]
    orig_shape = sio.imread(dataset.image_paths[0]).shape[:2]
    print(shape, orig_shape)
    mask_patch = torch.ones(len(perspective_transforms), *pert_shape).float()
    # initial sign is just the sign in the last image of
    # the sequence (because it is usually the biggest)
    print("len(dataset):",len(dataset))
    print("patch_transforms.shape:", patch_transforms.shape)
    print("dsize:",pert_shape[1:])
    perturbation = kornia.warp_perspective(
        next(
            iter(
                data.DataLoader(
                    DataSequence(
                        args.seqpath, transform=Compose([ToPILImage(), ToTensor()])
                    ),
                    batch_size=len(dataset),
                )
            )
        ),
        patch_transforms,
        dsize=pert_shape[1:],
    )[-1:]
    # can try blurring it if you want
    # perturbation = torch.rand_like(mask_patch[0])[None]
    # perturbation = torch.ones_like(mask_patch[0])[None] / 2
    # perturbation = torch.zeros_like(mask_patch[0])[None]
    # perturbation = torch.ones_like(mask_patch[0])[None]
    # perturbation = kornia.filters.max_blur_pool2d(perturbation, 3, ceil_mode=True)
    # perturbation = kornia.filters.median_blur(perturbation, (3, 3))
    perturbation = kornia.resize(perturbation, pert_shape[1:])
    num_iters = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        blurred_pert = kornia.resize(blurred_pert, perturbation.shape[2:])

        imgs = next(iter(data_loader))
        perturbation_warp = kornia.warp_perspective(
            torch.vstack([blurred_pert for _ in range(len(perspective_transforms))]),
            perspective_transforms,
            dsize=orig_shape,
        )
        warp_masks = kornia.warp_perspective(
            mask_patch, perspective_transforms, dsize=orig_shape
        )
        warp_masks = kornia.resize(warp_masks, shape)
        perturbation_warp = kornia.resize(perturbation_warp, shape)
        imgs = imgs * (1 - warp_masks)
        imgs += perturbation_warp * warp_masks
        save_image(imgs, sample_dir / f"pert_imgs_{i}.png")
        save_image(perturbation, sample_dir / f"pert_{i}.png")
        # print("imgs.shape:", imgs.shape)
        print(f"{imgs.shape=}")
        y = model(imgs.to(device))
        if args.direction == "left":
            # loss = F.mse_loss(y, torch.full_like(y, 1.5))  # left
            loss = F.mse_loss(y, torch.full_like(y, -1.0))  # left
        else:
            # loss = F.mse_loss(y, torch.full_like(y, -1.5))  # right
            loss = F.mse_loss(y, torch.full_like(y, 1.0))  # right
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
        # perturbation = torch.clamp(
        #     perturbation - np.sign(perturbation.grad) / 100, 0, 1
        # )  # a variation of the fast gradient sign method
        smoothed_gradients = kornia.filters.gaussian_blur2d(
            0.5 * perturbation.grad / torch.linalg.norm(perturbation.grad),
            (5, 5),
            (3, 3),
        )  # smooth the gradients first to get less noisy billboards
        perturbation = torch.clamp(perturbation - smoothed_gradients, 0, 1)
        model.zero_grad()

    with torch.no_grad():
        perturbation = perturbation.detach()
        blurred_pert = perturbation
        blurred_pert = kornia.filters.median_blur(perturbation, (3, 3))
        blurred_pert = kornia.resize(blurred_pert, perturbation.shape[2:])
        imgs = next(iter(data_loader))
        perturbation_warp = kornia.warp_perspective(
            torch.vstack([blurred_pert for _ in range(len(perspective_transforms))]),
            perspective_transforms,
            dsize=orig_shape,
        )
        warp_masks = kornia.warp_perspective(
            mask_patch, perspective_transforms, dsize=orig_shape
        )
        warp_masks = kornia.resize(warp_masks, shape)
        perturbation_warp = kornia.resize(perturbation_warp, shape)
        perturbed_imgs = imgs * (1 - warp_masks)
        perturbed_imgs += perturbation_warp * warp_masks

        orig_angles = model(imgs.to(device))
        pert_angles = model(perturbed_imgs.to(device))
        print("Original outputs:", orig_angles)
        print("Adversarial outputs:", pert_angles)
        arrowed_imgs = []
        orig_arrowed_imgs = []
        for img, pert_img, a1, a2 in zip(
            imgs, perturbed_imgs, orig_angles, pert_angles
        ):
            img_processed = draw_arrows(img.cpu().numpy(), a1.cpu().numpy())
            orig_arrowed_imgs.append(img_processed)
            img_processed = draw_arrows(pert_img.cpu().numpy(), a1.cpu().numpy(), a2.cpu().numpy())
            arrowed_imgs.append(img_processed)
        save_image(
            torch.from_numpy(np.asarray(arrowed_imgs)), sample_dir / "arrows.png"
        )
        save_image(
            torch.from_numpy(np.asarray(orig_arrowed_imgs)),
            sample_dir / "arrows_orig.png",
        )


if __name__ == "__main__":
    main2(parse_args())
