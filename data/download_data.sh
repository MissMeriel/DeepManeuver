#!/usr/bin/bash

. ../.venv-dm/bin/activate
pip install --upgrade pip
pip install zenodo-get
zenodo_get 8284640
unzip DeepManeuver-paper-data.zip