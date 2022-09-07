# test phase
import torch
import os
from torch.autograd import Variable
from net import GhostFusion_net
import utils
from utils import gradient
from scipy.misc import imread, imsave, imresize
from args_fusion import args
import numpy as np
import time
import cv2


def load_model(path, input_nc, output_nc):

    nest_model = GhostFusion_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

    nest_model.eval()
    #nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
    # encoder
    en_r = model.encoder(img1)
    en_v = model.encoder(img2)
    f = model.fusion(en_r, en_v, strategy_type=strategy_type)
    img_fusion = model.decoder(f);
    return img_fusion[0]

def _generate_fusion_mf(model,imgBase1,imgBase2,imgDetail1,imgDetail2):
    imgGradDetail1 = gradient(imgDetail1);
    imgGradDetail2 = gradient(imgDetail2);    
    shape = imgBase1.shape;
    imgGradDetail1 = torch.abs(imgGradDetail1);
    imgGradDetail2 = torch.abs(imgGradDetail2);
    en_GradDetail1 = model.encoder(imgGradDetail1);
    en_GradDetail2 = model.encoder(imgGradDetail2);
    focusMap1 = model.fusion(en_GradDetail1,en_GradDetail2,strategy_type='AGL1')[0];
    ones = torch.ones(1,1,shape[2],shape[3]);
    if (args.cuda):
        ones = ones.cuda(args.device);    
    focusMap2 = ones-focusMap1;    

    fBase = imgBase1*focusMap1+imgBase2*focusMap2;
    fDetail = imgDetail1*focusMap1+imgDetail2*focusMap2;
    return fBase,fDetail

def run_demo(model, irBase_path,irDetail_path, visBase_path, visDetail_path, output_path_root, index, BS,DS, mode):

    irBase_img = utils.get_test_images(irBase_path, height=None, width=None, mode=mode)
    irDetail_img = utils.get_test_images(irDetail_path, height=None, width=None, mode=mode)
    visBase_img = utils.get_test_images(visBase_path, height=None, width=None, mode=mode)
    visDetail_img = utils.get_test_images(visDetail_path, height=None, width=None, mode=mode)
    
    if args.cuda:
        irBase_img = irBase_img.cuda(args.device)
        irDetail_img = irDetail_img.cuda(args.device)
        visBase_img = visBase_img.cuda(args.device)
        visDetail_img = visDetail_img.cuda(args.device)
        model = model.cuda(args.device);
    irBase_img = Variable(irBase_img, requires_grad=False)
    irDetail_img = Variable(irDetail_img, requires_grad=False)
    visBase_img = Variable(visBase_img, requires_grad=False)
    visDetail_img = Variable(visDetail_img, requires_grad=False)
    if (BS=='AGL1' and DS=='AGL1'):
        fusedBase,fusedDetail = _generate_fusion_mf(model,irBase_img,visBase_img,irDetail_img,visDetail_img);
    else:
        #strategy_type_list = strategy_type_list = ['AVG', 'L1','SC','MAX','AGL1']
        #Base L1
        fusedBase = _generate_fusion_image(model, BS, irBase_img, visBase_img)
        fusedDetail = _generate_fusion_image(model, DS, irDetail_img, visDetail_img)
    fusedBase = fusedBase[0].cpu();
    fusedBase = fusedBase.squeeze().squeeze();
    fusedBase = fusedBase.numpy();
    fusedBase = fusedBase*255;
    
    file_name = 'fuseBase'+str(index) + '.png'
    output_path = output_path_root + file_name
    
    imsave(output_path,fusedBase);
    
    #Detail max

    fusedDetail = fusedDetail[0].cpu();
    fusedDetail = fusedDetail.squeeze().squeeze();
    fusedDetail = fusedDetail.numpy();
    fusedDetail = fusedDetail*255;
   
    fusedDetail = fusedDetail - np.mean(fusedBase);
    
    file_name = 'fuseDetail'+str(index) + '.png'
    output_path = output_path_root + file_name
    
    imsave(output_path,fusedDetail);
    
    #finalFuseResult
    fusedFinalResult = fusedDetail + fusedBase;
    
    ############################ multi outputs ##############################################
    file_name = 'fuseMedical'+str(index) + '.png'
    output_path = output_path_root + file_name
    
    imsave(output_path,fusedFinalResult);
    print(output_path)


def main():

    test_path = "sample_input/"

    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['AVG', 'L1','SC','MAX','AGL1']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    BS = strategy_type_list[0];
    DS = strategy_type_list[3];
    output_path = './outputs/';

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = 1
    if in_c == 1:
        out_c = in_c
        mode = 'L'
        model_path = args.model_path_gray
    else:
        out_c = in_c
        mode = 'RGB'
        model_path = args.model_path_rgb

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[2])
        ssim_weight_str = args.ssim_path[2]
        model = load_model(model_path, in_c, out_c)
        for i in range(1): 
            index = i + 1
            irBase_path = test_path + 'Medical_IRBase.bmp'
            irDetail_path = test_path + 'Medical_IRDetail.bmp'
            visBase_path = test_path + 'Medical_VISBase.bmp'
            visDetail_path = test_path + 'Medical_VISDetail.bmp'
            run_demo(model,irBase_path,irDetail_path, visBase_path, visDetail_path, output_path, index, BS, DS, mode)
    print('Done......')

if __name__ == '__main__':
    main()
