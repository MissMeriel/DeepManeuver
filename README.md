# DeepManeuver Perturbation Generator

## Installation

DeepManeuver relies on Python 3.8, BeamNGpy, and the BeamNG driving simulator.

Download Python 3.8 [here](https://www.python.org/downloads/release/python-380/).
Download CUDA Toolkit 11.6 [here](https://developer.nvidia.com/cuda-11-6-0-download-archive)

Activate a virtual environment and install the dependencies for this project:
```bash
python3.8 -m venv .venv
. .venv/bin/activate
pip install requirements.txt
```

Install torch with cuda enabled [here](https://pytorch.org/get-started/locally/).
Backup options if the linked instructions do not work:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116 # nightly build
```

Request a research license from BeamNG [here](https://support.beamng.com/) and install BeamNG.research.v1.7.0.1.
They're based in Germany so there is a time difference.
If you don't receive a response in a timely fashion email them at [research@beamng.gmbh](mailto:research@beamng.gmbh).

## Run DeepManeuver and output performance metrics

```bash
python collect-perturbed-trace-metas-demo1.py <path-to-external-dependencies> <road-id>
```

## Run DeepBillboard Alone

```bash
python -m deepbillboard dave.onnx sequences/Digital/digital_Udacity_straight1/ --direction=right
```

This will output results to `./samples/digital_Udacity_straight1`. 
The file `arrows.png` shows all of the images in the sequence with modified signs, and with arrows representing the predicted steering angles (blue for original, green for adversarial).
`pert_i.png` shows what the sign should look like after iteration `i`, and `pert_imgs_i.png` shows the images in the sequence with the modified sign at iteration `i`.

## Sequences

## Setup References

You may need to install [Visual Studio 2015, 2019 and 2019 redistributable](https://support.microsoft.com/en-nz/help/2977003/the-latest-supported-visual-c-downloads) to run on Windows 10.
See [Matplotlib issue #18292](https://github.com/matplotlib/matplotlib/issues/18292/).

Install torch with cuda enabled [here](https://pytorch.org/get-started/locally/).