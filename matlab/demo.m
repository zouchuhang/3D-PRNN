% demo to run cuboid fitting
% "prim_gt" is the predicted primitive, with the format "[mesh number, shape, transition, rotation]"
% If fitting fails due to random optimization, prim_gt is all zero
%
% To fit multiple primitives, see "script_parse_primitive.m", "script_parse_primitive_symmetry.m" and "script_refine_parse_primitive.m"

addpath(genpath('./minFunc_2012/'));

% gt mesh
vox_data = load(['../data/ModelNet10_mesh/chair/modelnet_chair_train.mat']);
vox_data = vox_data.voxTile;

gt_savename = 'Mygt.mat';
voxel_scale = 30;
sample_grid = 7;

gt_num = 100;

gt_voxel = reshape(vox_data(gt_num,:,:,:), voxel_scale, voxel_scale, voxel_scale);
%gt_voxel = permute(gt_voxel, [3,2,1]);
x_gt = []; y_gt = []; z_gt = [];
x_gt_inv = []; y_gt_inv = []; z_gt_inv = [];
for i = 1:voxel_scale
    % positive
    [x_gt_tmp, y_gt_tmp] = find(gt_voxel(:,:,i)); % define Qj
    x_gt = [x_gt;x_gt_tmp];
    y_gt = [y_gt;y_gt_tmp];
    z_gt = [z_gt;i*ones(size(x_gt_tmp))];
    % negative
    [x_gt_tmp, y_gt_tmp] = find(~gt_voxel(:,:,i)); % define Qj
    x_gt_inv = [x_gt_inv;x_gt_tmp];
    y_gt_inv = [y_gt_inv;y_gt_tmp];
    z_gt_inv = [z_gt_inv;i*ones(size(x_gt_tmp))];
end
% positive
gt_pt = [x_gt, y_gt, z_gt];
gt.gt_pt = gt_pt;
% negative
gt_pt_inv = [x_gt_inv, y_gt_inv, z_gt_inv];
gt.gt_pt_inv = gt_pt_inv;

%optimization
options = [];
options.display = 'none';
options.MaxIter = 100;
options.Method = 'lbfgs';
options.LS_init = 2;

rng('shuffle');
trans = randperm(size(gt_pt,1)); % random initialization
trans = gt_pt(trans(1),:);
shape = rand(1,3)*voxel_scale/3+3;
if shape(3) + trans(3) > 25
    trans(3) = trans(3) - (shape(3) + trans(3)-25);
end
if shape(2) + trans(2) > 25
    trans(2) = trans(2) - (shape(2) + trans(2)-25);
end
if shape(1) + trans(1) > 25
    trans(1) = trans(1) - (shape(1) + trans(1)-25);
end
save(gt_savename,'gt');

prim_gt = zeros(1, 19);
prim_gt(1,1:10) = [gt_num, shape, trans, 0,0,0];

Rv = [0;0;0];
theta = 0;
affv = prim_gt(1,2:7);
stop_thres = 1e-1;
cnt = 1;

while 1
    tic
    x = minFunc(@sampleEijOpt,[affv';Rv],options);
    %keyboard
    if norm(x(1:6) - affv')  < stop_thres % converge
        break
    end
    affv = x(1:6)';
    elapse = toc;
    fprintf('%d opt: %fs\n', cnt, elapse);
    % quit wrong result
    shape = x(1:3); trans = x(4:6)';
    if sum(shape<0) > 0 || shape(1)*shape(2)*shape(3) > 27000
        fprintf('shape < 0\n');
        break
    end
    sample_dist_x = shape(1)/sample_grid;% sampling 
    sample_x = 0:sample_dist_x:shape(1);
    sample_dist_y = shape(2)/sample_grid;
    sample_y = 0:sample_dist_y:shape(2);
    sample_dist_z = shape(3)/sample_grid;
    sample_z = 0:sample_dist_z:shape(3);
    [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
    sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];

    % refinement on rotation
    Rv = Rv/(theta+eps);
    [loss, sample_pt_dst] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv', theta, gt);
    loss_max = loss;
    theta_max = theta;
    Rv_max = Rv';
    Rv_r = [1,0,0];
    for i = -0.54:0.06:0.54
        if norm(Rv'-Rv_r) < 0.5
            theta_r = theta + i;
        else
            theta_r = i;
        end
        [loss_r, sample_pt_dst_r] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
        if loss_r > loss_max
            loss_max = loss_r;
            theta_max = theta_r;
            Rv_max = Rv_r;
            sample_pt_dst = sample_pt_dst_r;
        end
    end
    Rv_r = [0,1,0];
    for i = -0.54:0.18:0.54
        if norm(Rv'-Rv_r) < 0.5
            theta_r = theta + i;
        else
            theta_r = i;
        end
        [loss_r, sample_pt_dst_r] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
        if loss_r > loss_max
            loss_max = loss_r;
            theta_max = theta_r;
            Rv_max = Rv_r;
            sample_pt_dst = sample_pt_dst_r;
        end
    end
    Rv_r = [0,0,1];
    for i = -0.54:0.18:0.54
        if norm(Rv'-Rv_r) < 0.5
            theta_r = theta + i;
        else
            theta_r = i;
        end
        [loss_r, sample_pt_dst_r] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
        if loss_r > loss_max
            loss_max = loss_r;
            theta_max = theta_r;
            Rv_max = Rv_r;
            sample_pt_dst = sample_pt_dst_r;
        end
    end
    Rv = Rv_max*theta_max;
    Rv = Rv';
    %theta = theta_max;
    if cnt > 4
        break
    end
    cnt = cnt +1;
end

if sum(shape<0) > 0 || shape(1)*shape(2)*shape(3) > 27000
    prim_gt(1,11:end) = zeros(1,9)
else
    x = minFunc(@sampleEijOpt2,[x(1:6);Rv],options);
    prim_gt(1,11:end) = [ x(1:6)' Rv']
end
