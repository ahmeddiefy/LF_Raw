clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = './data/TrainingData/Test';
mkdir('test');


list = dir([sceneFolder, '/*', []]);
dataNames = setdiff({list.name}, {'.', '..'});
dataPaths = strcat(strcat(sceneFolder, '/'), dataNames);
numDatasets = length(dataNames);


for nd = 1:numDatasets
    dataName = dataNames{nd};
    data_path = dataPaths{nd};
	list = dir([data_path, '/*', []]);
	sceneNames = setdiff({list.name}, {'.', '..'});
	scenePaths = strcat(strcat(data_path, '/'), sceneNames);
	numScenes = length(sceneNames);
    mkdir(strcat('./data/test/',dataName));
    mkdir(strcat('./data/test/',dataName,'/in'));
    mkdir(strcat('./data/test/',dataName),'/gt');
    for ns = 1:numScenes
        sceneName = sceneNames{ns};
        numImgsX = 14;
        numImgsY = 14;

        resultPath = [scenePaths{ns}];

        inputImg = im2double(imread(resultPath));
        inputImg = rgb2ycbcr(inputImg);

        h = size(inputImg, 1) / numImgsY;
        w = size(inputImg, 2) / numImgsX;

        fullLF = zeros(h, w, 3, numImgsY, numImgsX);

        for ax = 1 : numImgsX
            for ay = 1 : numImgsY
                fullLF(:, :, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
            end
        end


        fullLF = fullLF(1:h, 1:w, :, 5:11, 5:11);


        img_raw = zeros(h*7, w*7, 3);



        for ax = 1 : 7
            for ay = 1 : 7
                img_raw(ay:7:end, ax:7:end, :) = fullLF(:, :, :, ay, ax);

            end
        end




        s = zeros(h,w,3,3,3);


        s(:,:,:,1,1) = fullLF(:,:,:,1,1);
        s(:,:,:,1,2) = fullLF(:,:,:,1,4);
        s(:,:,:,1,3) = fullLF(:,:,:,1,7);

        s(:,:,:,2,1) = fullLF(:,:,:,4,1);
        s(:,:,:,2,2) = fullLF(:,:,:,4,4);
        s(:,:,:,2,3) = fullLF(:,:,:,4,7);

        s(:,:,:,3,1) = fullLF(:,:,:,7,1);
        s(:,:,:,3,2) = fullLF(:,:,:,7,4);
        s(:,:,:,3,3) = fullLF(:,:,:,7,7);


        step_img = zeros(h*3, w*3, 3);


        for ax = 1 : 3
            for ay = 1 : 3
                step_img(ay:3:end, ax:3:end, :) = s(:, :, :, ay, ax);

            end
        end

        img = imresize(step_img,[h*7, w*7],'nearest');


        gt = single(img_raw);
        in = single(img);







        patch_name = strcat('./data/test/',dataName,'/gt/', sceneName(1:end-4));
        save(patch_name, 'gt');
        patch_name = strcat('./data/test/',dataName,'/in/', sceneName(1:end-4));
        save(sprintf(patch_name), 'in');

    end
    

    
    
end
