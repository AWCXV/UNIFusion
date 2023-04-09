# UNIFusion
This is the code of the paper titled as "UNIFusion: A Lightweight Unified Image Fusion Network".

## Usage
- Download all these files.
- If you want to test your own images on our model, "matlab_code_for_creating_base_and_detail_layers/main.m" is ready for you to generate the base and detail layers.

All the necessary parameter settings can be found at "args_fusion.py".

### Testing
We provide a series of testing files for different fusion tasks.

*e.g.* Infrared and visible image fusion task
Run the following code:
```
python test_imageTNO.py
```
The fusion results will be presented at the "outputs" folder.

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

## Contact Informaiton
If you have any questions, please contact me at <chunyang_cheng@163.com>.

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
