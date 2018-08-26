clc;
clear;
% demo for 6 primitives on one chair
addpath(genpath('./Voxel Plotter/'))
addpath(genpath('inhull'));
%prim_dir = '../draw3D-master/';
vox_data = load(['../data/ModelNet10_mesh/chair/modelnet_chair_train.mat']);
vox_data = vox_data.voxTile;
mesh_data = load(['../data/ModelNet10_mesh/chair/mesh_train.mat']);
mesh_data = mesh_data.meshTile;
addpath(genpath('./minFunc_2012/'))
save_file = './mn_chair_train_prim1/'; % means to refine on the 1st primitive

% prim_2
savename =  [save_file 'primall_gt_iter1_ref.mat'];
gt_savename = 'Mygt.mat'; % already enacode in opt
voxel_scale = 30;
sample_grid = 7;
visual = 0;
ntrain = size(vox_data, 1);

% number of primitive to refine
prim_i = 1;

% previous result, load all other primitives except the 1st pirmitve, since we want to refine on the first primitive
%prim1 = load('./prim_all/mn_chair_train_prim1/primall_prim1_new.mat');
%prim1 = prim1.prim_all;
%prim_set{1} = prim1;
prim1 = load('./prim_all/mn_chair_train_prim2/primall_prim1_new.mat');
prim1 = prim1.prim_all;
prim_set{1} = prim1;
prim1 = load('./prim_all/mn_chair_train_prim3/primall_prim1_new.mat');
prim1 = prim1.prim_all;
prim_set{2} = prim1;
prim1 = load('./prim_all/mn_chair_train_prim4/primall_prim1_new.mat');
prim1 = prim1.prim_all;
prim_set{3} = prim1;
prim1 = load('./prim_all/mn_chair_train_prim5/primall_prim1_new.mat');
prim1 = prim1.prim_all;
prim_set{4} = prim1;
prim1 = load('./prim_all/mn_chair_train_prim6/primall_prim1_new.mat');
prim1 = prim1.prim_all;
prim_set{5} = prim1;
prim1 = load('./prim_all/mn_chair_train_prim7/primall_prim1_new.mat');
prim1 = prim1.prim_all;
prim_set{6} = prim1;

% symmetry
%prim1_prev = load('./prim_all/mn_chair_train_prim1/primall_prim1_new_sym.mat');
%prim1_prev = prim1_prev.prim_all;
%prim_set_sym{1} = prim1_prev;
prim1_prev = load('./prim_all/mn_chair_train_prim2/primall_prim1_new_sym.mat');
prim1_prev = prim1_prev.prim_all;
prim_set_sym{1} = prim1_prev;
prim1_prev = load('./prim_all/mn_chair_train_prim3/primall_prim1_new_sym.mat');
prim1_prev = prim1_prev.prim_all;
prim_set_sym{2} = prim1_prev;
prim1_prev = load('./prim_all/mn_chair_train_prim4/primall_prim1_new_sym.mat');
prim1_prev = prim1_prev.prim_all;
prim_set_sym{3} = prim1_prev;
prim1_prev = load('./prim_all/mn_chair_train_prim5/primall_prim1_new_sym.mat');
prim1_prev = prim1_prev.prim_all;
prim_set_sym{4} = prim1_prev;
prim1_prev = load('./prim_all/mn_chair_train_prim6/primall_prim1_new_sym.mat');
prim1_prev = prim1_prev.prim_all;
prim_set_sym{5} = prim1_prev;
prim1_prev = load('./prim_all/mn_chair_train_prim7/primall_prim1_new_sym.mat');
prim1_prev = prim1_prev.prim_all;
prim_set_sym{6} = prim1_prev;

% current refined result
prim1 = load('./prim_all/mn_chair_train_prim1/primall_prim1_new_re.mat');
prim1 = prim1.prim_all;

prim_gt = zeros(ntrain, 19);

for vox_num_q = ntrain
    % gt
    fprintf('Process data# %d\n', vox_num_q)
    gt_voxel = reshape(vox_data(vox_num_q,:,:,:), voxel_scale, voxel_scale, voxel_scale);
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
    gt_thres = size(gt_pt,1)*0.04;
    % sample
    if visual 
        figure(1)
        [shape_vis]=VoxelPlotter(gt_voxel,1,1); 
        view(3);axis equal;hold on; scatter3(0,0,0);
        keyboard
    end
    prim_gt(vox_num_q,1:10) = prim1(vox_num_q,1:10);

    new_pt = [];
    for pn = 1:numel(prim_set)
        if pn == prim_i
            shape = prim1(vox_num_q,11:13);
            trans = prim1(vox_num_q,14:16);
            Rv = prim1(vox_num_q,17:19);
            theta = prim1(vox_num_q,20);
            % put back current gt points
            in_p = get_inhall_pt(shape, trans, Rv, theta, gt_pt, 0.5);
            Cidx = find(in_p);
            new_pt = gt.gt_pt(Cidx,:);
            if visual
                figure(2)
                scatter3(new_pt(:,1),new_pt(:,2),new_pt(:,3));
                keyboard
            end
        end
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
        % knock out prev sample
        in_p = get_inhall_pt(shape_prev, trans_prev, Rv_prev, theta_prev, gt_pt, 0);
        prev_pt = gt_pt(in_p,:);
        if visual
            figure(2)
            scatter3(prev_pt(:,1),prev_pt(:,2),prev_pt(:,3),color(pn));
            keyboard
        end
        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(find(in_p),:) = [];
        gt_pt(find(in_p),:) = [];

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
        in_p = get_inhall_pt(shape_prev, trans_prev, Rv_prev, theta_prev, gt_pt, 0);
        prev_pt = gt_pt(in_p,:);
        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(find(in_p),:) = [];
        gt_pt(find(in_p),:) = [];
    end

    % put back points
    if ~isempty(new_pt)
        Cidx = find(~ismember(new_pt,gt.gt_pt,'rows'));
        gt.gt_pt = [gt.gt_pt;new_pt(Cidx,:)];
    end
    if visual
        figure(2)
        scatter3(gt.gt_pt(:,1),gt.gt_pt(:,2),gt.gt_pt(:,3));
        figure(1)
        keyboard
    end

    % optimization
    rng('shuffle');
    % optimization
    options = [];
    options.display = 'none';
    options.MaxIter = 100;
    options.Method = 'lbfgs';
    options.LS_init = 2;
    save(gt_savename,'gt');
    shape = prim1(vox_num_q,11:13);
    trans = prim1(vox_num_q,14:16);
    Rv = prim1(vox_num_q,17:19);
    theta = prim1(vox_num_q,20);
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
        figure(1)
        scatter3(sample_pt_dst(:,1), sample_pt_dst(:,2), sample_pt_dst(:,3));
        keyboard;
    end
    cnt = 1;
    affv = [shape trans];
    Rv = Rv';
    stop_thres = 1e-1;
    [loss_ori, ~] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv', theta, gt);
    Rv = Rv * theta;
    Rv_ori = Rv;
    trans_ori = trans;
    shape_ori = shape;
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
            [sample_pt_dst] = sample_core_Eij_sum_only(sample_pt, trans, [1,1,1], Rv/(theta+eps), theta);
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
        %theta = theta_max;
        if cnt > 4
            break
        end
        cnt = cnt +1;
    end
    if sum(shape<0) > 0 || shape(1)*shape(2)*shape(3) > 27000
        prim_gt(vox_num_q,11:end) = [shape_ori, trans_ori, Rv_ori'];
    else
        x = minFunc(@sampleEijOpt2,[x(1:6);Rv],options);
        shape = x(1:3)';
        trans = x(4:6)';
        theta = norm(Rv)*sign(sum(Rv));
        sample_dist_x = shape(1)/sample_grid;% sampling 
        sample_x = 0:sample_dist_x:shape(1);
        sample_dist_y = shape(2)/sample_grid;
        sample_y = 0:sample_dist_y:shape(2);
        sample_dist_z = shape(3)/sample_grid;
        sample_z = 0:sample_dist_z:shape(3);
        [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
        sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];
        [loss_new, ~] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv/(theta+eps)', theta, gt);
        %keyboard
        if loss_new > loss_ori
            prim_gt(vox_num_q,11:end) = [x(1:6)' Rv'];
        else
            prim_gt(vox_num_q,11:end) = [shape_ori, trans_ori, Rv_ori'];
        end
    end
    if mod(vox_num_q,10) == 0
        save(savename, 'prim_gt', '-v7.3');
    end
end
save(savename, 'prim_gt', '-v7.3');
