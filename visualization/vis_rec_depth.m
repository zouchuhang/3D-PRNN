% visualize generation samples

cls = 'night_stand';
type = 'test';

ts_rep = 10;
ts_depth_rep = 5;

vox_data = load(['../data/ModelNet10_mesh/' cls '/modelnet_' cls  '_test.mat']);
vox_data = vox_data.voxTile;
%vox_data_tr = load(['ModelNet10_mesh/' cls '/modelnet_' cls  '_train.mat']);
%vox_data_tr = vox_data_tr.voxTile;
mesh_data = load(['../data/ModelNet10_mesh/' cls '/mesh_test.mat']);
obj_mesh = mesh_data;

color = {'red','green','blue','magenta','cyan','yellow', 'black', 'white','black','white','black','white','black'};
 
dp_ts = load(['../data/depth_map/depth_mn_test_' cls '_ts.mat']);
% gt
ts_prim = load(['../data/prim_gt/prim_sort_mn_' cls '_test.mat'],'primset');

% res
res = load(['../data/sample_generation/test_res_mn_' cls '_prob.mat']);
res = res.x;

VolumeSize = 30;

for start_num = 1:ts_rep:size(res,1)/4
    prim_num = dp_ts.match_id(ceil(ceil(start_num/ts_rep)/ts_depth_rep),2);
    disp(prim_num)
    vox_gt = reshape(vox_data(prim_num,:,:,:),30,30,30);
    %vox_gt = permute(vox_gt, [2,1,3]);
    x_gt = []; y_gt = []; z_gt = [];
    x_gt_inv = []; y_gt_inv = []; z_gt_inv = [];
    for j = 1:30
        % positiveprim
        [x_gt_tmp, y_gt_tmp] = find(vox_gt(:,:,j)); % define Qj
        x_gt = [x_gt;x_gt_tmp];
        y_gt = [y_gt;y_gt_tmp];
        z_gt = [z_gt;j*ones(size(x_gt_tmp))];
        % negative
        [x_gt_tmp, y_gt_tmp] = find(~vox_gt(:,:,j)); % define Qj
        x_gt_inv = [x_gt_inv;x_gt_tmp];
        y_gt_inv = [y_gt_inv;y_gt_tmp];
        z_gt_inv = [z_gt_inv;j*ones(size(x_gt_tmp))];
    end
    % gt pt
    gt_pt = [x_gt, y_gt, z_gt];
    % neg pt
    gt_pt_inv = [x_gt_inv, y_gt_inv, z_gt_inv];
    
    primset = ts_prim.primset{prim_num};
    vertices = [];
    faces = [];
    cnt = 1;
    
    
for re_num = start_num+3:start_num+3 %start_num + ts_rep-1
    
        res_prim = res((re_num-1)*4+1:(re_num-1)*4+2, :);
        res_rot = res((re_num-1)*4+3, :);
        res_sym = res(re_num*4, :);
        stop_idx = find(res_prim(1,:) == 0,1,'first');
        % check each result
    
        tol_cnt = 1;
        for res_row = 1:3:stop_idx-3
            prim_rot = res_rot(res_row:res_row+2);
            prim_r = [res_prim(1, res_row:res_row+2) res_prim(2, res_row:res_row+2) prim_rot];
            sym_r = res_sym(res_row:res_row+2);
            
            shape = prim_r(:,1:3);
            trans = prim_r(:,4:6);
            Rv = prim_r(:,7:9);
            [Rv_y, Rv_i] = max(abs(Rv));
            theta = Rv(Rv_i);
            %[Rv_y, Rv_i] = max(sym_r);
            Rv = zeros(1,3); Rv(Rv_i) = 1;
            %theta = prim_rot(Rv_i);
            
            %keyboard
            %if 15 - prim_r(4)- prim_r(1)/2> 3 % sum(sym_r > 0.1)>0
            if prim_r(4)+prim_r(1) < 15
                prim_r(4) = 28 - prim_r(4)-prim_r(1)/2;
                
                shape = prim_r(:,1:3);
                trans = prim_r(:,4:6);
                [~, Rv_t] = max(abs(Rv));
                if Rv_t ~=1
                    theta = -theta;
                end
                
            end
            tol_cnt = tol_cnt + 1;
        end
    
end  

%figure(3)
%[shape_vis]=VoxelPlotter(vox_gt,1,1); hold on
%view(3); axis equal;scatter3(0,0,0);
% if visual
figure(1)
subplot(1,4,1); 
imagesc(reshape(dp_ts.depth_tile(start_num,:,:),64,64));
axis image; title('input depth');
subplot(1,4,2); title('gt voxel')% voxel
cla;
[shape_vis]=VoxelPlotter(vox_gt,1,1);
hold on; view(3); axis equal;scatter3(0,0,0);
subplot(1,4,3); title('gt primitive')% gt
cla;
for j = 1:size(primset.ori,1)
        prim_r = primset.ori(j,:);
        prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
        prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
        prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
        prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
        prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
        prim_pt_mean = mean(prim_pt);
    
        %[Rv_y, Rv_i] = max(sym_r);
        Rv = prim_r(17:19);
        theta = prim_r(20);
        vx = getVX(Rv);% rotation
        Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
        prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
        prim_pt = prim_pt*Rrot;
        prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
        prim_pt = bsxfun(@min,prim_pt,30);

        vertices = prim_pt;
        faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
        light('Position',[-1 -1 0],'Style','local')
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);

        view(3); axis equal; hold on;
        axis([0,30,0,30,0,30])
        
        prim_r = primset.sym(j,:);
        
        prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
        prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
        prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
        prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
        prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
        prim_pt_mean = mean(prim_pt);
    
        [Rv_y, Rv_i] = max(sym_r);
        Rv = prim_r(17:19);
        theta = prim_r(20);
        vx = getVX(Rv);% rotation
        Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
        prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
        prim_pt = prim_pt*Rrot;
        prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
        prim_pt = bsxfun(@min,prim_pt,30);
        
        vertices = prim_pt;
        light('Position',[-1 -1 0],'Style','local') 
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
        view(3); axis equal; hold on;
        axis([0,30,0,30,0,30])
        
end


subplot(1,4,4); title('generation')% pred
hold off;cla;
tol_cnt = 1;
res_prim = res((start_num-1)*4+1:(start_num-1)*4+2, :);
res_rot = res((start_num-1)*4+3, :);
res_sym = res(start_num*4, :);
stop_idx = find(res_prim(1,:) == 0,1,'first');

pred_vertices = [];
pred_faces = [];
cnt = 1;

for res_row = 1:3:stop_idx-3
            prim_rot = res_rot(res_row:res_row+2);
            prim_r = [res_prim(1, res_row:res_row+2) res_prim(2, res_row:res_row+2) prim_rot];
            sym_r = res_sym(res_row:res_row+2); 
        
            prim_pt_x = [0 prim_r(1) prim_r(1) 0 0 prim_r(1) prim_r(1) 0];
            prim_pt_y = [0 0 prim_r(2) prim_r(2) 0 0 prim_r(2) prim_r(2)];
            prim_pt_z = [0 0 0 0 prim_r(3) prim_r(3) prim_r(3) prim_r(3)];
            prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
            prim_pt = bsxfun(@plus, prim_pt, prim_r(:,4:6));
            prim_pt_mean = mean(prim_pt);
    
            Rv = prim_r(:,7:9);
            [Rv_y, Rv_i] = max(abs(Rv));
            theta = Rv(Rv_i);
            %[Rv_y, Rv_i] = max(sym_r);
            Rv = zeros(1,3); Rv(Rv_i) = 1;
            %theta = prim_rot(Rv_i);
            
            vx = getVX(Rv);% rotation
            Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
            prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
            prim_pt = prim_pt*Rrot;
            prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
            vertices = prim_pt;
            faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
            
            %figure
            %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
            light('Position',[-1 -1 0],'Style','local');
            patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
            
            
            %if  15 - prim_r(4)- prim_r(1)/2> 3 %sum(sym_r > 0.5)>1.5
            if prim_r(4)+prim_r(1) < 15
                prim_r(4) = 30 - prim_r(4)-prim_r(1)/2;
                prim_pt_x = [0 prim_r(1) prim_r(1) 0 0 prim_r(1) prim_r(1) 0];
                prim_pt_y = [0 0 prim_r(2) prim_r(2) 0 0 prim_r(2) prim_r(2)];
                prim_pt_z = [0 0 0 0 prim_r(3) prim_r(3) prim_r(3) prim_r(3)];
                prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
                prim_pt = bsxfun(@plus, prim_pt, prim_r(:,4:6));
                prim_pt_mean = mean(prim_pt);
    
                [~, Rv_t] = max(abs(Rv));
                if Rv_t ~=1
                    theta = -theta;
                end
                %Rv = [0 0 0]; theta = 0;
                vx = getVX(Rv);% rotation
                Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
                prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
                prim_pt = prim_pt*Rrot;
                prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
                %prim_pt = bsxfun(@min,prim_pt,30);
                vertices = prim_pt;
                %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
                patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
                
            end
            tol_cnt = tol_cnt + 1;
        view(3); axis equal; hold on;
        axis([0,30,0,30,0,30])
        %keyboard
end
        
keyboard
end

