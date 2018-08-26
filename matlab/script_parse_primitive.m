clc;
clear;
addpath(genpath('./Voxel Plotter/'))
addpath(genpath('inhull'));
vox_data = load(['../data/ModelNet10_mesh/chair/modelnet_chair_train.mat']);
vox_data = vox_data.voxTile;
mesh_data = load(['../data/ModelNet10_mesh/chair/mesh_train.mat']);
mesh_data = mesh_data.meshTile;
addpath(genpath('./minFunc_2012/'))
save_file = './mn_chair_train_prim1/'; % to save 1st parsed primitive

% prim_2a
savename =  [save_file 'primall_gt_iter1.mat'];
gt_savename = 'Mygt.mat'; % already enacode in opt
voxel_scale = 30;
sample_grid = 7;
visual = 0;
ntrain = size(vox_data, 1);

% previous result, once you parsed out and saved the previous primitives, start the script to parse the next, load previously saved results
prim_set = {};
%prim1 = load('./prim_all/mn_chair_train_prim1/primall_prim1_new_re.mat');
%prim1 = prim1.prim_all;
%prim_set{1} = prim1;

% symmetry
prim_set_sym = {};
%prim1 = load('./prim_all/mn_chair_train_prim1/primall_prim1_new_sym_re.mat');
%prim1 = prim1.prim_all;
%prim_set_sym{1} = prim1;

prim_gt = zeros(ntrain, 19);

for vox_num_q = 1:ntrain
    % gt
    fprintf('Process data# %d\n', vox_num_q)
    gt_voxel = reshape(vox_data(vox_num_q,:,:,:), voxel_scale, voxel_scale, voxel_scale);
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
    gt_thres = size(gt_pt,1)*0.03;
    % sample
    if visual 
        figure(1)
        [shape_vis]=VoxelPlotter(gt_voxel,1,1); 
        view(3);axis equal;hold on; scatter3(0,0,0);
        keyboard
    end
    for pn = 1:numel(prim_set)
        prim_param = prim_set{pn};
        shape_prev = prim_param(vox_num_q,11:13);
        trans_prev = prim_param(vox_num_q,14:16);
        Rv_prev = prim_param(vox_num_q,17:19); %theta_prev = norm(Rv_prev);
        theta_prev = prim_param(vox_num_q,20); 
        if visual
            sample_dist_x = shape_prev(1)/sample_grid;
            sample_x = 0:sample_dist_x:shape_prev(1);
            sample_dist_y = shape_prev(2)/sample_grid;
            sample_y = 0:sample_dist_y:shape_prev(2);
            sample_dist_z = shape_prev(3)/sample_grid;
            sample_z = 0:sample_dist_z:shape_prev(3);
            [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
            sample_pt_prev = [sample_X(:), sample_Y(:), sample_Z(:)];
            [sample_pt_dst_prev] = sample_core_Eij_sum_only(sample_pt_prev, trans_prev, [1,1,1], Rv_prev, theta_prev);
            figure(1);
            scatter3(sample_pt_dst_prev(:,1),sample_pt_dst_prev(:,2),sample_pt_dst_prev(:,3));
            scatter3(0, 0, 0);
            keyboard
        end
        in_p = get_inhall_pt(shape_prev, trans_prev, Rv_prev, theta_prev, gt_pt, 0.5);
        prev_pt = gt_pt(in_p,:);

        Cidx = find(in_p);
        if visual
            figure(2)
            scatter3(prev_pt(:,1),prev_pt(:,2),prev_pt(:,3),color(pn));
            keyboard
        end
        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(Cidx,:) = [];
        gt_voxel(gt_pt(Cidx,1), gt_pt(Cidx,2), gt_pt(Cidx,3)) = 0;
        gt_pt(Cidx,:) = [];

        % knock out sym if exist
        prim_param = prim_set_sym{pn};
        if prim_param(vox_num_q,1) == 0
            continue
        end
        shape_prev = prim_param(vox_num_q,11:13);
        trans_prev = prim_param(vox_num_q,14:16);
        Rv_prev = prim_param(vox_num_q,17:19); %theta_prev = norm(Rv_prev);
        theta_prev = prim_param(vox_num_q,20);
        if visual
            sample_dist_x = shape_prev(1)/sample_grid;
            sample_x = 0:sample_dist_x:shape_prev(1);
            sample_dist_y = shape_prev(2)/sample_grid;
            sample_y = 0:sample_dist_y:shape_prev(2);
            sample_dist_z = shape_prev(3)/sample_grid;
            sample_z = 0:sample_dist_z:shape_prev(3);
            [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
            sample_pt_prev = [sample_X(:), sample_Y(:), sample_Z(:)];
            [sample_pt_dst_prev] = sample_core_Eij_sum_only(sample_pt_prev, trans_prev, [1,1,1], Rv_prev, theta_prev);
            figure(1);
            scatter3(sample_pt_dst_prev(:,1),sample_pt_dst_prev(:,2),sample_pt_dst_prev(:,3), 'b');
            scatter3(0, 0, 0);
            pause(1)
            keyboard
        end
        % knock out prev sample
        in_p = get_inhall_pt(shape_prev, trans_prev, Rv_prev, theta_prev, gt_pt, 0.5);
        prev_pt = gt_pt(in_p,:);

        Cidx = find(in_p);
        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(Cidx,:) = [];
        gt_voxel(gt_pt(Cidx,1), gt_pt(Cidx,2), gt_pt(Cidx,3)) = 0;
        gt_pt(Cidx,:) = [];
    end
    if size(gt_pt,1) < gt_thres
        continue
    end
    if visual
        figure(2)
        scatter3(gt.gt_pt(:,1),gt.gt_pt(:,2),gt.gt_pt(:,3));
        figure(1)
        keyboard
    end
    
    % optimization
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
    prim_gt(vox_num_q,1:10) = [vox_num_q, shape, trans, 0,0,0];
    % optimization
    options = [];
    options.display = 'none';
    options.MaxIter = 100;
    options.Method = 'lbfgs';
    options.LS_init = 2;
    
    % sample points
    save(gt_savename,'gt');

    if visual
        sample_dist_x = shape(1)/sample_grid;% sampling 
        sample_x = 0:sample_dist_x:shape(1);
        sample_dist_y = shape(2)/sample_grid;
        sample_y = 0:sample_dist_y:shape(2);
        sample_dist_z = shape(3)/sample_grid;
        sample_z = 0:sample_dist_z:shape(3);
        [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
        sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];
        [sample_pt_dst] = sample_core_Eij_sum_only(sample_pt, trans, [1,1,1], [0 0 0], 0);
        figure(1)
        scatter3(sample_pt_dst(:,1), sample_pt_dst(:,2), sample_pt_dst(:,3));
        keyboard;
    end
    cnt = 1;
    Rv = [0;0;0];
    theta = 0;
    affv = prim_gt(vox_num_q,2:7);
    stop_thres = 1e-1;
    IoUthres = 0.55;
     while 1
        tic
        x = minFunc(@sampleEijOpt,[affv';Rv],options);
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

        if visual
            [sample_pt_dst] = sample_core_Eij_sum_only(sample_pt, trans, [1,1,1], Rv, theta);
            scatter3(sample_pt_dst(:,1), sample_pt_dst(:,2), sample_pt_dst(:,3));
            keyboard
        end

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
        if visual
            scatter3(sample_pt_dst(:,1), sample_pt_dst(:,2), sample_pt_dst(:,3));
            keyboard
        end
        %keyboard
        Rv = Rv_max*theta_max;
        Rv = Rv';
        if cnt > 4
            break
        end
        cnt = cnt +1;
    end
    if sum(shape<0) > 0 || shape(1)*shape(2)*shape(3) > 27000
        prim_gt(vox_num_q,11:end) = zeros(1,9);
    else
        x = minFunc(@sampleEijOpt2,[x(1:6);Rv],options);
        prim_gt(vox_num_q,11:end) = [x(1:6)' Rv'];
    end
    if mod(vox_num_q,10) == 0
        save(savename, 'prim_gt', '-v7.3');
    end
end
save(savename, 'prim_gt', '-v7.3');
