# Data Preparation Impact on Semantic Segmentation of 3D Mobile LiDAR Point Clouds Using Deep Neural Networks

Installing pre-requisites:

* Install `python` --This repo is tested with `{3.8}`
* Install `torch` --This repo is tested with `{1.5.0}` and `cuda 10.2`
* Install dependencies: `pip3 install -r requirements.txt`
* Compile the C++ extension modules for python located in `./KPConv/cpp_wrappers`. Open a terminal in this folder, and run:

          sh compile_wrappers.sh
