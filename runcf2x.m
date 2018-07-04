

%   The code is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.
%
%   Hierarchical Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015
%
% Contact:
%   Chao Ma (chaoma99@gmail.com), or
%   Jia-Bin Huang (jbhuang1@illinois.edu).


clc;
addpath('utility','model','external/matconvnet/matlab');
vl_setupnn();

% Note that the default setting does not enable GPU
% TO ENABLE GPU, recompile the MatConvNet toolbox  
%vl_compilenn();
base_path   = '/opt/dataset/otb100/';
	if ispc(), base_path = strrep(base_path, '\', '/'); end
	if base_path(end) ~= '/', base_path(end+1) = '/'; end
	
	%list all sub-folders
	contents = dir(base_path);
	names = {};
	for k = 1:numel(contents),
		name = contents(k).name;
		if isdir([base_path name]) && ~any(strcmp(name, {'.', '..'})),
			names{end+1} = name;  %#ok
		end
    end
    sigma=0.1;
    for lyi=1:length(sigma)
%         clear net;
        tscale=[1];
        for ti=1:1
%             sigma(lyi)
            for k=1:length(names)
           run_tracker(names{k}, 0, 0,sigma(lyi));
            end
        end
    end
