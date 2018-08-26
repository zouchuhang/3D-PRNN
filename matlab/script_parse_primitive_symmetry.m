% check symmetry

addpath(genpath('./Voxel Plotter/'))
addpath(genpath('inhull'));
vox_data = load(['../data/ModelNet10_mesh/chair/modelnet_chair_train.mat']);
vox_data = vox_data.voxTile;
mesh_data = load(['../data/ModelNet10_mesh/chair/mesh_train.mat']);
mesh_data = mesh_data.meshTile;
addpath(genpath('./minFunc_2012/'))
opt_file = './mn_chair_train_prim1/';

% prim_2
savename = [opt_file 'primall_gt_iter1_sym.mat'];
gt_savename = 'Mygt.mat'; % already encode in opt
voxel_scale = 30;
sample_grid = 7;
visual = 0;
IoUthres = 0.3;
ntrain = size(vox_data, 1);

prim_gt = load(['./prim_all/' opt_file '/primall_gt_iter1.mat']);
prim_gt = prim_gt.prim_all;

% previous result, once you parsed out and saved the previous primitives, start the script to parse the next, load previously saved results
prim_set_sym = {};
%prim1 = load('./prim_all/mn_chair_train_prim1/primall_prim1_new_sym.mat');
%prim1 = prim1.prim_all;
%prim_set_sym{1} = prim1;

prim_set_prev = {};
%prim1 = load('./prim_all/mn_chair_train_prim1/primall_prim1_new.mat');
%prim1 = prim1.prim_all;
%prim_set_prev{1} = prim1;

prim1_sym = zeros(size(prim_gt));

for vox_num_q = 1:size(prim_gt,1)
    disp(vox_num_q)
    if prim_gt(vox_num_q,1) == 0
        continue
    end

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
    % if symmetry
    cen = mean(gt_pt);
    
    % if result is symmetry
    shape = prim_gt(vox_num_q,11:13);
    trans = prim_gt(vox_num_q,14:16);
    Rv = prim_gt(vox_num_q,17:19);
    theta = prim_gt(vox_num_q,20);
    shape_cen = shape/2+trans;
    if shape_cen(1) < cen(1)-3 || shape_cen(1) > cen(1)+3
        if visual
                 figure(1)
        [shape_vis]=VoxelPlotter(gt_voxel,1,1); 
        view(3);axis equal;hold on; scatter3(0,0,0);
        end
    for pn = 1:numel(prim_set_prev)
        prim_param = prim_set_prev{pn};
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
            %keyboard
        end
        in_p = get_inhall_pt(shape_prev, trans_prev, Rv_prev, theta_prev, gt_pt, 0.5);
        prev_pt = gt_pt(in_p,:);
        Cidx = find(in_p);

        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(Cidx,:) = [];
        
        gt_pt(Cidx,:) = [];
        % knock out sym if exist
        prim_param = prim_set_sym{pn};
        if prim_param(vox_num_q,1) == 0
            continue
        end
        shape_prev = prim_param(vox_num_q,11:13);
        trans_prev = prim_param(vox_num_q,14:16);
        Rv_prev = prim_param(vox_num_q,17:19); 
        %theta_prev = norm(Rv_prev);
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
        in_p = get_inhall_pt(shape_prev, trans_prev, Rv_prev, theta_prev, gt_pt, 0.5);
        prev_pt = gt_pt(in_p,:);
        Cidx = find(in_p);
        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(Cidx,:) = [];
        gt_voxel(gt_pt(Cidx,1), gt_pt(Cidx,2), gt_pt(Cidx,3)) = 0;
        gt_pt(Cidx,:) = [];
         if visual
            figure(2)
            scatter3(gt_pt(:,1),gt_pt(:,2),gt_pt(:,3),color(pn));
            keyboard
        end
    end
    
        in_p = get_inhall_pt(shape, trans, Rv, theta, gt_pt, 0);
        prev_pt = gt_pt(in_p,:);
        Cidx = find(in_p);
        gt.gt_pt_inv = [gt.gt_pt_inv; prev_pt];
        gt_pt_inv = [gt_pt_inv; prev_pt];
        gt.gt_pt(Cidx,:) = [];
        gt_voxel(gt_pt(Cidx,1), gt_pt(Cidx,2), gt_pt(Cidx,3)) = 0;
        gt_pt(Cidx,:) = [];
         if visual
            figure(2)
            scatter3(gt_pt(:,1),gt_pt(:,2),gt_pt(:,3),color(pn));
            keyboard
        end
        sample_dist_x = shape(1)/sample_grid;
        sample_x = 0:sample_dist_x:shape(1);
        sample_dist_y = shape(2)/sample_grid;
        sample_y = 0:sample_dist_y:shape(2);
        sample_dist_z = shape(3)/sample_grid;
        sample_z = 0:sample_dist_z:shape(3);
        [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
        sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];

        [sample_pt_dst_prev] = sample_core_Eij_sum_only(sample_pt, trans, [1,1,1], Rv, theta);
        
        if visual
            figure(1)
        scatter3(sample_pt_dst_prev(:,1),sample_pt_dst_prev(:,2),sample_pt_dst_prev(:,3));
        end
        % take sym
        trans(1) = cen(1) + cen(1) - shape_cen(1);
        trans(1) = trans(1) - shape(1)/2;
        if Rv(3) ~= 0 || Rv(2) ~= 0
            theta = -theta;
        end
        if visual
        [sample_pt_dst] = sample_core_Eij_sum_only(sample_pt, trans, [1,1,1], Rv, theta);
        scatter3(sample_pt_dst(:,1),sample_pt_dst(:,2),sample_pt_dst(:,3));
        keyboard
        end
        % refine
                stop_thres = 1e-1;
         options = [];
    options.display = 'none';
    options.MaxIter = 100;
    options.Method = 'lbfgs';
    options.LS_init = 2;
    save(gt_savename,'gt');
    cnt = 1;
    affv = [shape, trans];
    [loss_ori, sample_pt_dst] = sample_core_Eij_sum3(sample_pt, trans, shape, [1,1,1], Rv, theta, gt);
    shape_ori = shape;
    trans_ori = trans;
    if 0
        while 1
        % refinement on rotation
        [loss, sample_pt_dst] = sample_core_Eij_sum3(sample_pt, trans, shape, [1,1,1], Rv, theta, gt);
        loss_max = loss;
        theta_max = theta;
        Rv_max = Rv;
        Rv_r = [1,0,0];
        for i = -0.54:0.06:0.54
            if norm(Rv-Rv_r) == 0
                theta_r = theta + i;
            else
                theta_r = i;
            end
            [loss_r, sample_pt_dst_r] = sample_core_Eij_sum3(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
            if loss_r > loss_max
                loss_max = loss_r;
                theta_max = theta_r;
                Rv_max = Rv_r;
                sample_pt_dst = sample_pt_dst_r;
            end
        end
        Rv_r = [0,1,0];
        for i = -0.54:0.18:0.54
            if norm(Rv-Rv_r) == 0
                theta_r = theta + i;
            else
                theta_r = i;
            end
            [loss_r, sample_pt_dst_r] = sample_core_Eij_sum3(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
            if loss_r > loss_max
                loss_max = loss_r;
                theta_max = theta_r;
                Rv_max = Rv_r;
                sample_pt_dst = sample_pt_dst_r;
            end
        end
        Rv_r = [0,0,1];
        for i = -0.54:0.18:0.54
            if norm(Rv-Rv_r) == 0
                theta_r = theta + i;
            else
                theta_r = i;
            end
            [loss_r, sample_pt_dst_r] = sample_core_Eij_sum3(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
            if loss_r > loss_max
                loss_max = loss_r;
                theta_max = theta_r;
                Rv_max = Rv_r;
                sample_pt_dst = sample_pt_dst_r;
            end
        end
        %if visual
        %    scatter3(sample_pt_dst(:,1), sample_pt_dst(:,2), sample_pt_dst(:,3));
        %    keyboard
        %end
        %keyboard
        Rv = Rv_max*theta_max;
        Rv = Rv';
        tic
        x = minFunc(@sampleEijOpt2,[affv';Rv],options);
        shape = x(1:3); 
        if sum(shape<0) > 0 || shape(1)*shape(2)*shape(3) > 27000
            fprintf('shape < 0\n');
            break
        end
        if norm(x(1:6) - affv')  < stop_thres % converge
            %Rv = Rv*theta;
            break
        end
        affv = x(1:6)';
        elapse = toc;
        fprintf('%d opt: %fs\n', cnt, elapse);
        if cnt > 4
            break
        end
        cnt = cnt +1;
        theta = norm(Rv)*sign(sum(Rv));
        Rv = Rv'/theta;
        end
    end

        x = minFunc(@sampleEijOpt2,[affv';Rv'*theta],options);

        if visual
        shape = x(1:3)';
        trans = x(4:6)';
        %sample_dist_x = shape(1)/sample_grid;
        sample_x = 0:1/6:1;%0:sample_dist_x:shape(1);
        %sample_dist_y = shape(2)/sample_grid;
        sample_y = 0:1/6:1;%0:sample_dist_y:shape(2);
        %sample_dist_z = shape(3)/sample_grid;
        sample_z = 0:1/6:1;%0:sample_dist_z:shape(3);
        [sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
        sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];
        sample_pt = bsxfun(@times, sample_pt,shape);
        [sample_pt_dst_prev] = sample_core_Eij_sum_only(sample_pt, trans, [1,1,1], Rv', theta);
        if visual
        scatter3(sample_pt_dst_prev(:,1),sample_pt_dst_prev(:,2),sample_pt_dst_prev(:,3));
        keyboard
        end
        end
        shape = x(1:3)'; 
        trans = x(4:6)';

        prim_pt_x = [0 shape(1) shape(1) 0 0 shape(1) shape(1) 0];
        prim_pt_y = [0 0 shape(2) shape(2) 0 0 shape(2) shape(2)];
        prim_pt_z = [0 0 0 0 shape(3) shape(3) shape(3) shape(3)];
        prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
        prim_pt = bsxfun(@plus, prim_pt, trans);
        prim_pt_mean = mean(prim_pt);
    
        vx = getVX(Rv);% rotation
        Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
        prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
        prim_pt = prim_pt*Rrot;
        prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
        prim_pt = bsxfun(@min,prim_pt,30);
        vertices = prim_pt;
        in_p = inhull(gt_pt,vertices);
        in_n = inhull(gt_pt_inv,vertices);
        IoU = sum(in_p)/(sum(in_p)+sum(in_n));
        if sum(shape<0) > 0 || shape(1)*shape(2)*shape(3) > 27000 || IoU < IoUthres
            %prim1_sym(vox_num_q,:) = zeros(1,19);
        else
            trans = x(4:6);
            [loss_new, sample_pt_dst] = sample_core_Eij_sum3(sample_pt, trans', shape', [1,1,1], Rv, theta, gt);
            if loss_new > loss_ori
                    prim1_sym(vox_num_q,:) = [prim_gt(vox_num_q,1:10) x(1:6)' Rv*theta theta];
            else
                    prim1_sym(vox_num_q,:) = [prim_gt(vox_num_q,1:10) shape_ori trans_ori Rv*theta theta];
                disp('ori is better !')
            end
        end
    end
    if mod(vox_num_q,10) == 0
        save(savename, 'prim1_sym', '-v7.3');
    end
end
save(savename, 'prim1_sym', '-v7.3');
