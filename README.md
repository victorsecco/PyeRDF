# PhD
This repository basically is most of the programming code I developed during my PhD


# Installation (with Conda)

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/victorsecco/PhD.git
   cd PhD
Alternatively, download the ZIP from GitHub and extract it, then cd into the extracted folder.


2. **Create and activate a Conda environment**


    ```bash
    conda create -n epdf python=3.9 -y
    conda activate epdf

Python versions >=3.9 will work.

3. **Build the package**


    ```bash
    python -m pip install --upgrade pip build
    python -m build

This will create distribution files (.whl and .tar.gz) inside the dist/ directory.

3. **Install the package**

    ```bash
    pip install dist/phd-0.1.0-py3-none-any.whl

4. **Install additional dependencies**

    ```bash
    pip install opencv-python tifffile matplotlib pandas scipy medpy

# Installation (with Pip)

Since this installation include C compiled dependencies (numpy, scipy, matplotlib), conda is more suitable for installation. If only using pip, python versions above 3.11, that have the ready-made wheels for these packages, are needed. To use only wheels and not built any dependencies from source code, add ```bash --only-binary=:all:``` to the bash command. 

    ```bash
    pip install --only-binary=:all: "numpy<2" "matplotlib<3.9" "pandas<2.2" "scipy<1.12" tifffile opencv-python