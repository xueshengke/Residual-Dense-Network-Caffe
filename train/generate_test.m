clear; close all;
addpath util;
%% settings
folder = 'data/validation';
savepath = 'test.h5';
scale = 4;  
size_input = 24; 
size_label = scale * size_input; % output = (input - 1) * stride + kernel - 2 * padding
stride = size_input;
batch_size = 2;

%% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
file_bmp = dir(fullfile(folder, '*.bmp'));
file_png = dir(fullfile(folder, '*.png'));
filepaths = [file_bmp; file_png];

for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder, filepaths(i).name));
    image = im2double(image);
    
    im_label = modcrop(image, scale);
    im_input = imresize(im_label, 1/scale, 'bicubic');
    [hei, wid, channel] = size(im_input);
    
    for x = 1 : stride : hei - size_input + 1
        for y = 1 : stride : wid - size_input + 1

            locx = scale * (x - 1) + 1;
            locy = scale * (y - 1) + 1;
%             locx = floor(scale * (x + (size_input - 1)/2) - (size_label + scale)/2 + 1);
%             locy = floor(scale * (y + (size_input - 1)/2) - (size_label + scale)/2 + 1);
            
            subim_input = im_input(x : x + size_input - 1, y : y + size_input - 1, :);
            subim_label = im_label(locx : locx + size_label - 1, locy : locy + size_label - 1, :);
            
            count = count + 1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
        end
    end
end
order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = batch_size;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read = (batchno-1)*chunksz;
    batchdata = data(:, :, :, last_read+1:last_read+chunksz); 
    batchlabs = label(:, :, :, last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
