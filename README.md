# UNIFusion
This is the code of the paper titled as "UNIFusion: A Lightweight Unified Image Fusion Network".

## Usage
- Download all these files.
- If you want to test your own images on our model, "matlab_code_for_creating_base_and_detail_layers/main.m" is ready for you to generate the base and detail layers.

### For IR/VIS, Multi-Exposure and Medical image fusion tasks:
*eg*. Infrared and visible image fusion task
1. Change the paths in "test_imageTNO.py" to run our model on your own infrared and visible image pairs.
2. run the "test_imageTNO.py" by using the following code:
```
python test_imageTNO.py
```

### For multi-focus image fusion task:
1. Change the paths in "test_imageMF.py" to run our model on your own near-focused and far-focused image pairs.
2. run the "test_imageMF.py", you can customize the *k1* and *k2* values in the file "fusion_strategy.py" (Marked with corresponding notes).

You can find all the other parameter settings in the "args_fusion.py" file.

## Training
Training dataset can be found at this website: https://pjreddie.com/projects/coco-mirror/

Put the images at the "train2014" folder.

```
python train.py
```

## Environment
- Python 3.7.3
- torch 1.7.1
- scipy 1.2.0

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
