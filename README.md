# UNIFusion
This is the code of the paper titled as "UNIFusion: A Lightweight Unified Image Fusion Network".

## Usage
1. Download all these files (folder "data" is not mandatory, you can use your own testing data).

### For infrared and visible image fusion task:
1. Change the paths in "test_imageTNO.py" to run our model on your own infrared and visible image pairs.
2. run the "test_imageTNO.py" by using the following code:
```
python test_imageTNO.py
```

You can find all the other parameter settings in the "args_fusion.py" file.

## Environment
Python 3.7.3

## Updates:
⚡【2022-5-3】 the testing code for the TNO dataset and related parameter setting config files are available now, you can customize the paths in "test_imageTNO.py" to test your own images.

⚡【2022-4-29】 the matlab code for generating the base and detail layers are available at the folder of "matlab_code_for_creating_base_and_detail_layers". Please customize the paths information in the "main.m" file to obtain the expected images.

⚡【2022-4-28】 the input images (base and detail layers) are available at the folder of "/data".

## Acknowledgement
Most code of this implementation is based on the DenseFuse: https://github.com/hli1221/densefuse-pytorch

# Citation
If this work is helpful to you, please cite it as:
```
@article{cheng2021unifusion,
  title={UNIFusion: A Lightweight Unified Image Fusion Network},
  author={Cheng, Chunyang and Wu, Xiao-Jun and Xu, Tianyang and Chen, Guoyang},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={70},
  pages={1--14},
  year={2021},
  publisher={IEEE}
}
```
