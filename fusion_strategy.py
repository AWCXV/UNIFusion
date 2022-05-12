import torch
import math
import numpy as np
from utils import gradient2,sumPatch,sumPatch2
import torch.nn.functional as F
from scipy.misc import imread, imsave, imresize
from skimage import morphology
from args_fusion import args

EPSILON = 1e-10


def AVGfusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2
    
def MAXfusion(tensor1, tensor2):
    return torch.max(tensor1, tensor2);

def MINfusion(tensor1, tensor2):
    return torch.min(tensor1, tensor2);

def L1Fusion(tensor1, tensor2):
    print('Using L1 fusion strategy...');
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = f_spatial
    return tensor_f
   
def AGL1Fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size() 
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)   
    h = shape[2];
    w = shape[3];
    
    # ****************customize the k1 value here************
    spatial1 = sumPatch(spatial1,3);
    spatial2 = sumPatch(spatial2,3);
                        
    spatial1 = torch.gt(spatial1,spatial2);
    
    
    spatial1[0,0,:,:] = torch.from_numpy(morphology.remove_small_holes(spatial1[0,0,:,:].cpu().numpy(), area_threshold=h*w*0.05, connectivity=2, in_place=False));
    if (args.cuda):
        spatial1 = spatial1.cuda(args.device);
    spatial1 = spatial1.float();
    
    # ****************customize the k2 value here************
    spatial1 = sumPatch(spatial1,12);    
    return spatial1; 
    
  
  
def SCFusion(tensor1,tensor2):
    f_spatial = spatial_fusion(tensor1, tensor2);
    f_channel = channel_fusion(tensor1, tensor2);
    a = 0;
    print('Using SC fusion strategy and a='+str(a)+'...');
    tensor_f = a*f_spatial + (1-a)*f_channel;
    return tensor_f;
    
def channel_fusion(tensor1, tensor2):
    shape = tensor1.size()
    global_p1 = channel_attention(tensor1)
    global_p2 = channel_attention(tensor2)

    global_p_w1 = global_p1 / (global_p1+global_p2+EPSILON)
    global_p_w2 = global_p2 / (global_p1+global_p2+EPSILON)

    global_p_w1 = global_p_w1.repeat(1,1,shape[2],shape[3])
    global_p_w2 = global_p_w2.repeat(1,1,shape[2],shape[3])

    tensorf = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensorf    

def channel_attention(tensor, pooling_type = 'avg'):
    shape = tensor.size()
    global_p = F.avg_pool2d(tensor,kernel_size=shape[2:])
    return global_p

def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial




