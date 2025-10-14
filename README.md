# PyeRDF
This repo aims to provide a data processing framework for electron diffraction data to obtain the ePDF.

# Installation

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/victorsecco/PyeRDF.git
   
Alternatively, download the ZIP from GitHub and extract it, then cd into the extracted folder.


2. **Create and activate a Conda environment**


    ```bash
    conda create -n epdf python=3.12.8 -y
    conda activate epdf

I only tested the code in this python version. You can also just install python 3.12.8.

3. **Build the package**


    ```bash
    python -m pip install --upgrade pip build
    python -m build

This will create distribution files (.whl and .tar.gz) inside the dist/ directory.

3. **Install the package**

    ```bash
    pip install dist/phd-0.1.0-py3-none-any.whl

This should install all the dependencies and finish the installation. 

