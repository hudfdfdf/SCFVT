% GET_FEATURES: Extracting hierachical convolutional features

function feat = get_features(im, cos_window, layers)

global net
global enableGPU
% global net1;

if isempty(net)
    initial_net(layers(1));
  
% net1 = load(fullfile('model', 'imagenet-vgg-verydeep-19.mat'));
% net=net1;
% net.layers(layers(1)+1:end) = [];
% else
%     net=net1;
%     net.layers(layers(1)+1:end) = [];
% 
% % Switch to GPU mode
% 
% if enableGPU
%     net = vl_simplenn_move(net, 'gpu');
% end
end

sz_window = size(cos_window);

% Preprocessing
img = single(im);        % note: [0, 255] range
img = imResample(img, net.normalization.imageSize(1:2));
img = img - net.normalization.averageImage;
if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(net,img);

% Initialize feature maps
feat = cell(length(layers), 1);

for ii = 1:length(layers)
    
    % Resize to sz_window
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
%     for si=1:512  imshow(mat2gray(x(:,:,si))); pause(0.01);end
     x=x(:,:,1:4:end);    %4
%     xp=permute(x,[3,1,2]);
%     ti=1;
%     [m1,m2,m3]=size(xp);
%     for xpi=1:4:m1
%         xs(ti,1:m2,1:m3)=sum(xp(xpi:xpi+3,:,:),1);
%         ti=ti+1;
%     end
%     x=permute(xs,[2,3,1]);
    x = imResample(x, sz_window(1:2));
    
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    feat{ii}=x;

end

end
