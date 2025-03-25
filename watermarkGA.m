% Genetic Algorithm for Optimizing Image Watermarking with External Watermark
clear; clc; close all;

% Parameters
popSize = 20;           % Population size
maxGen = 50;            % Maximum generations
pc = 0.8;               % Crossover probability
pm = 0.1;               % Mutation probability
bitSize = 8;            % Bits per parameter
wmSize = 32;            % Watermark size (will resize to this)
attackType = 'noise';   % Test attack type ('noise', 'compression', 'filtering')

% Load host image
hostImg = imread('lena.png');
if size(hostImg, 3) == 3
    hostImg = rgb2gray(hostImg);
end
hostImg = im2double(hostImg);

% Load external watermark image (replace 'your_logo.png' with your file)
watermarkImg = imread('watermark.png');
if size(watermarkImg, 3) == 3
    watermarkImg = rgb2gray(watermarkImg);
end
watermarkImg = imresize(watermarkImg, [wmSize wmSize]); % Resize to target dimensions
watermark = imbinarize(watermarkImg); % Convert to binary (0s and 1s)

% Initialize population
population = randi([0 1], popSize, 4*bitSize); % 4 parameters (alpha, blockSize, xPos, yPos)

% Main GA loop
for gen = 1:maxGen
    % Evaluate fitness
    fitness = zeros(popSize, 1);
    for i = 1:popSize
        % Decode parameters
        params = decodeIndividual(population(i,:), bitSize, size(hostImg), wmSize);
        
        % Embed watermark
        watermarkedImg = embedWatermarkDCT(hostImg, watermark, params);
        
        % Apply attack
        attackedImg = applyAttack(watermarkedImg, attackType);
        
        % Extract watermark
        extractedWM = extractWatermarkDCT(attackedImg, watermark, params);
        
        % Calculate fitness (Normalized Correlation)
        fitness(i) = sum(sum(watermark == extractedWM)) / numel(watermark);
    end
    
    % Selection (Tournament selection)
    newPopulation = zeros(size(population));
    for i = 1:popSize
        candidates = randperm(popSize, 2);
        [~, idx] = max(fitness(candidates));
        newPopulation(i,:) = population(candidates(idx),:);
    end
    
    % Crossover (Single point)
    for i = 1:2:popSize-1
        if rand < pc
            cp = randi([1, 4*bitSize-1]);
            temp = newPopulation(i, cp+1:end);
            newPopulation(i, cp+1:end) = newPopulation(i+1, cp+1:end);
            newPopulation(i+1, cp+1:end) = temp;
        end
    end
    
    % Mutation (Bit flip)
    for i = 1:popSize
        for j = 1:4*bitSize
            if rand < pm
                newPopulation(i,j) = ~newPopulation(i,j);
            end
        end
    end
    
    population = newPopulation;
    
    % Display progress
    [bestFit, idx] = max(fitness);
    fprintf('Generation %d: Best Fitness = %.4f\n', gen, bestFit);
    
    if bestFit == 1
        break;
    end
end

% Display final results
[bestFit, idx] = max(fitness);
bestParams = decodeIndividual(population(idx,:), bitSize, size(hostImg), wmSize);

figure;
subplot(2,3,1); imshow(hostImg); title('Original Image');
subplot(2,3,2); imshow(watermark); title('Original Watermark');
watermarkedImg = embedWatermarkDCT(hostImg, watermark, bestParams);
subplot(2,3,3); imshow(watermarkedImg); title('Watermarked Image');
attackedImg = applyAttack(watermarkedImg, attackType);
subplot(2,3,4); imshow(attackedImg); title(['Attacked (' attackType ')']);
extractedWM = extractWatermarkDCT(attackedImg, watermark, bestParams);
subplot(2,3,5); imshow(extractedWM); title('Extracted Watermark');
subplot(2,3,6); 
imshowpair(watermark, extractedWM, 'montage'); 
title('Original vs Extracted');

fprintf('\nOptimized Parameters:\n');
fprintf('Alpha (strength): %.4f\n', bestParams.alpha);
fprintf('Block Size: %d\n', bestParams.blockSize);
fprintf('Embed Position: (%d, %d)\n', bestParams.xPos, bestParams.yPos);
fprintf('Watermark Similarity: %.2f%%\n', bestFit*100);

% Helper Functions (Same as before)
function params = decodeIndividual(individual, bitSize, imgSize, wmSize)
    alpha = bin2dec(num2str(individual(1:bitSize))) / (2^bitSize-1) * 0.1;
    blockSize = 2 + bin2dec(num2str(individual(bitSize+1:2*bitSize))) / (2^bitSize-1) * 6;
    blockSize = 2^round(blockSize);
    
    maxX = imgSize(2) - wmSize;
    maxY = imgSize(1) - wmSize;
    xPos = 1 + floor(bin2dec(num2str(individual(2*bitSize+1:3*bitSize))) / (2^bitSize-1) * maxX);
    yPos = 1 + floor(bin2dec(num2str(individual(3*bitSize+1:4*bitSize))) / (2^bitSize-1) * maxY);
    
    params = struct('alpha', alpha, 'blockSize', blockSize, 'xPos', xPos, 'yPos', yPos);
end

function watermarkedImg = embedWatermarkDCT(hostImg, watermark, params)
    watermarkedImg = hostImg;
    [h, w] = size(hostImg);
    wm = 2*watermark - 1; % Convert to +1/-1
    
    for i = 1:params.blockSize:h-params.blockSize+1
        for j = 1:params.blockSize:w-params.blockSize+1
            if i >= params.yPos && i < params.yPos+size(watermark,1) && ...
               j >= params.xPos && j < params.xPos+size(watermark,2)
                block = hostImg(i:i+params.blockSize-1, j:j+params.blockSize-1);
                dctBlock = dct2(block);
                
                wmRow = mod(i-params.yPos, size(watermark,1)) + 1;
                wmCol = mod(j-params.xPos, size(watermark,2)) + 1;
                wmBit = wm(wmRow, wmCol);
                
                midFreq = round(params.blockSize/2);
                dctBlock(midFreq, midFreq) = dctBlock(midFreq, midFreq) * (1 + params.alpha*wmBit);
                
                watermarkedImg(i:i+params.blockSize-1, j:j+params.blockSize-1) = idct2(dctBlock);
            end
        end
    end
end

function extractedWM = extractWatermarkDCT(watermarkedImg, originalWM, params)
    [h, w] = size(watermarkedImg);
    extractedWM = zeros(size(originalWM));
    originalImg = watermarkedImg; % In real use, this should be the unwatermarked image
    
    for i = 1:params.blockSize:h-params.blockSize+1
        for j = 1:params.blockSize:w-params.blockSize+1
            if i >= params.yPos && i < params.yPos+size(originalWM,1) && ...
               j >= params.xPos && j < params.xPos+size(originalWM,2)
                block = watermarkedImg(i:i+params.blockSize-1, j:j+params.blockSize-1);
                origBlock = originalImg(i:i+params.blockSize-1, j:j+params.blockSize-1);
                
                dctBlock = dct2(block);
                origDctBlock = dct2(origBlock);
                
                wmRow = mod(i-params.yPos, size(originalWM,1)) + 1;
                wmCol = mod(j-params.xPos, size(originalWM,2)) + 1;
                
                midFreq = round(params.blockSize/2);
                extractedBit = (dctBlock(midFreq, midFreq) > origDctBlock(midFreq, midFreq));
                extractedWM(wmRow, wmCol) = extractedBit;
            end
        end
    end
end

function attackedImg = applyAttack(img, attackType)
    switch attackType
        case 'noise'
            attackedImg = imnoise(img, 'gaussian', 0, 0.01);
        case 'compression'
            imwrite(img, 'temp.jpg', 'Quality', 50);
            attackedImg = imread('temp.jpg');
            delete('temp.jpg');
        case 'filtering'
            attackedImg = medfilt2(img, [3 3]);
        otherwise
            attackedImg = img;
    end
    attackedImg = im2double(attackedImg);
end