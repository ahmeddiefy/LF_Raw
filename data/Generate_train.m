clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = './data/TrainingData/Training';
mkdir('train');
patch_size = 128;
count = 1;
list = dir([data_path, '/*', []]);
sceneNames = setdiff({list.name}, {'.', '..'});
scenePaths = strcat(strcat(data_path, '/'), sceneNames);
numScenes = length(sceneNames);
for ns = 1:numScenes

    numImgsX = 14;
    numImgsY = 14;
    
    resultPath = [scenePaths{ns}];
    
    inputImg = im2double(imread(resultPath));
    inputImg = rgb2ycbcr(inputImg);
    inputImg = inputImg(:,:,1);

    h = size(inputImg, 1) / numImgsY;
    w = size(inputImg, 2) / numImgsX;
    
    fullLF = zeros(h, w, numImgsY, numImgsX);
    
    for ax = 1 : numImgsX
        for ay = 1 : numImgsY
            fullLF(:, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end);
        end
    end

    fullLF = fullLF(1:h, 1:w, 5:11, 5:11);
    



    
    gt = zeros(h*7, w*7);


    
    for ax = 1 : 7
        for ay = 1 : 7
            gt(ay:7:end, ax:7:end) = fullLF(:, :, ay, ax);

        end
    end

    input_LF = zeros(h,w,3,3,3);


    input_LF(:,:,:,1,1) = fullLF(:,:,:,1,1);
    input_LF(:,:,:,1,2) = fullLF(:,:,:,1,4);
    input_LF(:,:,:,1,3) = fullLF(:,:,:,1,7);

    input_LF(:,:,:,2,1) = fullLF(:,:,:,4,1);
    input_LF(:,:,:,2,2) = fullLF(:,:,:,4,4);
    input_LF(:,:,:,2,3) = fullLF(:,:,:,4,7);

    input_LF(:,:,:,3,1) = fullLF(:,:,:,7,1);
    input_LF(:,:,:,3,2) = fullLF(:,:,:,7,4);
    input_LF(:,:,:,3,3) = fullLF(:,:,:,7,7);

   
    
    step_img = zeros(h*3, w*3);

    
    for ax = 1 : 3
        for ay = 1 : 3
            step_img(ay:3:end, ax:3:end) = input_LF(:, :, ay, ax);

        end
    end
    
    in = single(imresize(step_img,[h*7, w*7],'nearest'));

    
    gt = single(gt);
    [H, W] = size(gt);

    
    for ix=1:floor(H/patch_size)
        for iy=1:floor(W/patch_size)
           patch_name = sprintf('./data/train/%d',count);
           gt_patch =  gt( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           in_patch= in( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           patch = gt_patch;
           save(patch_name, 'patch');
           patch = in_patch;
           save(sprintf('%s_1', patch_name), 'patch');
           count = count+1;
        end
    end

    
    display(count);
    
    
end
