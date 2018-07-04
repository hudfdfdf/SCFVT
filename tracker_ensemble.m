% tracker_ensemble: Correlation filter tracking with convolutional features
%
% Input:
%   - video_path:          path to the image sequence
%   - img_files:           list of image names
%   - pos:                 intialized center position of the target in (row, col)
%   - target_sz:           intialized target size in (Height, Width)
% 	- padding:             padding parameter for the search area
%   - lambda:              regularization term for ridge regression
%   - output_sigma_factor: spatial bandwidth for the Gaussian label
%   - interp_factor:       learning rate for model update
%   - cell_size:           spatial quantization level
%   - show_visualization:  set to True for showing intermediate results
% Output:
%   - positions:           predicted target position at each frame
%   - time:                time spent for tracking
%
%   It is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.
%
%   Hierarchical Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015
%
% Contact:
%   Chao Ma (chaoma99@gmail.com), or
%   Jia-Bin Huang (jbhuang1@illinois.edu).


function [positions, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization,video)


% ================================================================================
% Environment setting
% ================================================================================

indLayers = [35,33,28];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
nweights  = [1,1,0.5]; % Weights for combining correlation filter responses
% indLayers = [28];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
% nweights  = [1]; % Weights for combining correlation filter responses
numLayers = length(indLayers);

% Get image size and search window size
im_sz     = size(imread([video_path img_files{1}]));
window_sz = get_search_window(target_sz, im_sz, padding);

% Compute the sigma for the Gaussian function label
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;

%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size    d=bsxfun(@times,c,[1 2]);

l1_patch_num = floor(window_sz/ cell_size);

% Pre-compute the Fourier Transform of the Gaussian function label
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));

% Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
cos_window = hann(size(yf,1)) * hann(size(yf,2))';

% Create video interface for visualization
if(show_visualization)
    update_visualization = show_video(img_files, video_path);
end

% Initialize variables for calculating FPS and distance precision
time      = 0;
positions = zeros(numel(img_files), 2);
nweights  = reshape(nweights,1,1,[]);

% Note: variables ending with 'f' are in the Fourier domain.
model_xf     = cell(1, numLayers);
model_alphaf = cell(1, numLayers);
respo(1)=1;
apecs(1)=1;
% ================================================================================
% Start tracking
% ================================================================================
for frame = 1:numel(img_files),
    im = imread([video_path img_files{frame}]); % Load the image at the current frame
    if ismatrix(im)
        im = cat(3, im, im, im);
    end
%     if frame==293
%          frame
%     end
    tic();
    cnn(1)=0;
    % ================================================================================
    % Predicting the object position from the learned object model
    % ================================================================================
    if frame> 1
        % Extracting hierarchical convolutional features
        feat = extractFeature(im, pos, window_sz, cos_window, indLayers);
        % Predict position
        [pos,cnn(frame),respo(frame),apecs(frame)] = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
            model_xf, model_alphaf);
    end
    
    % ================================================================================
    % Learning correlation filters over hierarchical convolutional features
    % ================================================================================
    % Extracting hierarchical convolutional features
    if (frame==1) || mod(frame,6)==0;    %6
%      if (frame==1) ||(frame>=2 &&respo(frame)>mean(respo(2:frame))*0.4&&apecs(frame)>mean(apecs(2:frame))*0.6)
    feat  = extractFeature(im, pos, window_sz, cos_window, indLayers);
    % Model update
    [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
        model_xf, model_alphaf);
    end
    
    % ================================================================================
    % Save predicted position and timing
    % ================================================================================
    positions(frame,:) = pos;
    time = time + toc();
    
    % Visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        drawnow
        % 			pause(0.05)  % uncomment to run slower
    end
    % save
%     save_video=1;
%     if save_video,
%         rstidx=1;
%         pathdraw='/opt/123/CF2_result/';
%         pathsave=[pathdraw video '_' num2str(rstidx) '/'];
%         if ~exist(pathsave,'dir')
%             mkdir(pathsave)
%         end
%         imwrite(frame2im(getframe(gcf)),[pathsave num2str(frame) '.png']);
%     end
end
% save('/opt/123/CF2/result/cnn.mat','cnn');
end


function [pos,cnn,mres,apec] = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
    model_xf, model_alphaf)

% ================================================================================
% Compute correlation filter responses at each layer
% ================================================================================
res_layer = zeros([l1_patch_num, length(indLayers)]);

for ii = 1 : length(indLayers)
    zf = fft2(feat{ii});
    kzf=sum(zf .* conj(model_xf{ii}), 3) / numel(zf);
    res_layer(:,:,ii) = real(fftshift(ifft2(model_alphaf{ii} .* kzf)));  %equation for fast detection
% for ee=1:size(zf,3)
%     kzf=zf(:,:,ee) .* conj(model_xf{ii}(:,:,ee)) / numel(zf);
%      res(:,:,ee) = real(fftshift(ifft2(model_alphaf{ii} .* kzf)));  %equation for fast detection
% end
% res_layer2(:,:,ii)=sum(res,3);
% 
% sum(sum((res_layer(:,:,ii)-res_layer2(:,:,ii))>0.01))
end

% Combine responses from multiple layers (see Eqn. 5)
response = sum(bsxfun(@times, res_layer, nweights), 3);
cnn=max(response(:));
mres=max(response(:));
minres=min(response(:));
 apec=(mres-minres)^2/mean2((response-minres).^2);
% ================================================================================
% Find target location
% ================================================================================
% Target location is at the maximum response. we must take into
% account the fact that, if the target doesn't move, the peak
% will appear at the top-left corner, not at the center (this is
% discussed in the KCF paper). The responses wrap around cyclically.
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
vert_delta  = vert_delta  - floor(size(zf,1)/2);
horiz_delta = horiz_delta - floor(size(zf,2)/2);

% Map the position to the image space
pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];


end


function [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
    model_xf, model_alphaf)

numLayers = length(feat);

% ================================================================================
% Initialization
% ================================================================================
xf       = cell(1, numLayers);
alphaf   = cell(1, numLayers);

% ================================================================================
% Model update
% ================================================================================
for ii=1 : numLayers
    xf{ii} = fft2(feat{ii});
    kf = sum(xf{ii} .* conj(xf{ii}), 3) / numel(xf{ii});
    alphaf{ii} = yf./ (kf+ lambda);   % Fast training
end

% Model initialization or update
if frame == 1,  % First frame, train with a single image
    for ii=1:numLayers
        model_alphaf{ii} = alphaf{ii};
        model_xf{ii} = xf{ii};
    end
else
    % Online model update using learning rate interp_factor
    %if mod(frame,6)==0
    for ii=1:numLayers
        model_alphaf{ii} = (1 - interp_factor) * model_alphaf{ii} + interp_factor * alphaf{ii};
        model_xf{ii}     = (1 - interp_factor) * model_xf{ii}     + interp_factor * xf{ii};
    end
    %end
end


end

function feat  = extractFeature(im, pos, window_sz, cos_window, indLayers)

% Get the search window from previous detection
patch = get_subwindow(im, pos, window_sz);
% Extracting hierarchical convolutional features
feat  = get_features(patch, cos_window, indLayers);

end
