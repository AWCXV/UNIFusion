for ii=1:18
    %% Read the image
    VIS = im2double(imread(['../imagesMean/VIS/',num2str(ii),'.bmp']));
    IR = im2double(imread(['../imagesMean/IR/',num2str(ii),'.bmp']));
    %figure,imshow(VIS);
    %% Do the job
    alpha_t = .001;
    N_iter = 3;
    tic
    [VISBase,R] = muGIF(VIS,VIS,alpha_t,0,N_iter)  ;
    [IRBase,R] = muGIF(IR,IR,alpha_t,0,N_iter)  ;
    toc
    
    VISDetail = VIS - VISBase;
    IRDetail = IR - IRBase;
    
    VISDetail = VISDetail + mean(mean(VIS));
    IRDetail = IRDetail + mean(mean(IR));
    
    %figure,imshow(VISBase);
    %figure,imshow(VISDetail);
    
    imwrite(IRBase,['../imagesMean/IR/IRBase',num2str(ii),'.png']);
    imwrite(VISBase,['../imagesMean/VIS/VISBase',num2str(ii),'.png']);
    imwrite(IRDetail,['../imagesMean/IR/IRDetail',num2str(ii),'.png']);
    imwrite(VISDetail,['../imagesMean/VIS/VISDetail',num2str(ii),'.png']);
    %figure,imshow(T);
    disp(['Process:(',num2str(ii),'/21)']);
end

