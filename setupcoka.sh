#!/bin/bash

conda create --name oct python=3.9 -y
conda init bash
source ~/.bashrc
source activate oct

python -m pip install -r requirements.txt

mkdir attention_maps gan_images