function initial_net(ilayer)
% INITIAL_NET: Loading VGG-Net-19

global net;
net = load(fullfile('model', 'imagenet-vgg-verydeep-19.mat'));

% Remove the fully connected layers and classification layer
net.layers(ilayer+1:end) = [];

% Switch to GPU mode
global enableGPU;
if enableGPU
    net = vl_simplenn_move(net, 'gpu');
end

end
