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