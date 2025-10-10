# PhD
This repository basically is most of the programming code I developed during my PhD


Installation tutorial with conda

Access https://github.com/victorsecco/PhD

Download the .zip folder from the PhD Github, extract folder

open cmd, PowerShell or anaconda prompt. 

cd to the extracted folder

optional: create a venv, here I named the env epdf, but any name will do

conda create -n epdf 
conda activate epdf

conda install python=3.9

python -m pip install --upgrade build

python -m build

this will create a wheel file to install my packages, which will be created inside the dist folder

pip install /path/to/dist/phd-0.1.0-py3-none-any.whl

pip install opencv-python

pip install tifffile matplotlib pandas scipy medpy opencv-python