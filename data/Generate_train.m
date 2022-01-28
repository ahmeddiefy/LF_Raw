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
    



    
    img_raw = zeros(h*7, w*7);


    
    for ax = 1 : 7
        for ay = 1 : 7
            img_raw(ay:7:end, ax:7:end) = fullLF(:, :, ay, ax);

        end
    end

    ss = zeros(h,w,2,2);

    
    ss(:,:,1,1) = fullLF(:,:,1,1);
    ss(:,:,1,2) = fullLF(:,:,1,8);

    
    ss(:,:,2,1) = fullLF(:,:,8,1);
    ss(:,:,2,2) = fullLF(:,:,8,8);

   
    
    step_img = zeros(h*2, w*2);

    
    for ax = 1 : 2
        for ay = 1 : 2
            step_img(ay:2:end, ax:2:end) = ss(:, :, ay, ax);

        end
    end
    
    img_1 = single(imresize(step_img,[h*8, w*8],'nearest'));

    
    img_raw = single(img_raw);
    [H, W] = size(img_raw);

    
    for ix=1:floor(H/patch_size)
        for iy=1:floor(W/patch_size)
           patch_name = sprintf('./data/train/%d',count);
           img_raw_patch =  img_raw( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           img_1_patch= img_1( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           patch = img_raw_patch;
           save(patch_name, 'patch');
           patch = img_1_patch;
           save(sprintf('%s_1', patch_name), 'patch');
           count = count+1;
        end
    end

    
    display(count);
    
    
end
