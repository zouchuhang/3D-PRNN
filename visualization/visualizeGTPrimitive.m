% visualize gt primitive

addpath('./Voxel Plotter/');

cls =  'chair';
type = 'train';
visual = 1;

load(['../data/prim_gt/prim_sort_mn_' cls '_' type '.mat'],'primset');
%load(['prim_sort_mn_nightstand_' type '.mat'],'primset');

vox_data = load(['../data/ModelNet10_mesh/' cls '/modelnet_' cls '_' type '.mat']);
vox_data = vox_data.voxTile;
mesh_data = load(['../data/ModelNet10_mesh/' cls '/mesh_' type '.mat']);
obj_mesh = mesh_data;

color = {'red', 'green', 'blue', 'cyan', 'yellow', 'magenta','black','black','white','white', 'red', 'green', 'blue'};

for i = 1:numel(primset)
    disp(i);
    prim_num = i;
    vox_gt = reshape(vox_data(prim_num,:,:,:),30,30,30);
    figure(1)
    subplot(1,3,1)
    [shape_vis]=VoxelPlotter(vox_gt,1,1);
    hold on; view(3); axis equal;scatter3(0,0,0);
    subplot(1,3,2)
    [shape_vis]=VoxelPlotter(vox_gt,1,1);
    hold on; view(3); axis equal;scatter3(0,0,0);
    h3 = subplot(1,3,3);cla(h3);
    for j = 1:size(primset{i}.ori,1)
        prim_r = primset{i}.ori(j,:);
        prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
        prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
        prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
        prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
        prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
        prim_pt_mean = mean(prim_pt);
    
        Rv = prim_r(:,17:19);
        vx = getVX(Rv);% rotation
        theta = prim_r(:,20);
        Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
        prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
        prim_pt = prim_pt*Rrot;
        prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
        prim_pt = bsxfun(@min,prim_pt,30);

        vertices = prim_pt;
        faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
        subplot(1,3,2);
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'FaceAlpha',0.3);
        subplot(1,3,3);
        light('Position',[-1 -1 0],'Style','local')
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
        view(3); axis equal; hold on;
        axis([0,30,0,30,0,30])
        %figure(2)
        %light('Position',[5 100 15],'Style','local')
                %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
        %        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
        %        view(3); axis equal; hold on;
                figure(1)
        
        prim_r = primset{i}.sym(j,:);
        
        prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
        prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
        prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
        prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
        prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
        prim_pt_mean = mean(prim_pt);
    
        Rv = prim_r(:,17:19);
        vx = getVX(Rv);% rotation
        theta = prim_r(:,20);
        Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
        prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
        prim_pt = prim_pt*Rrot;
        prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
        prim_pt = bsxfun(@min,prim_pt,30);
        
        vertices = prim_pt;
        faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
        subplot(1,3,2);
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'FaceAlpha',0.3);
        subplot(1,3,3);
        light('Position',[-1 -1 0],'Style','local')
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
        view(3); axis equal; hold on;
        axis([0,30,0,30,0,30])
        %figure(2)
        %light('Position',[5 100 15],'Style','local')
                %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
        %        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
        %        view(3); axis equal; hold on;
                figure(1)
    end
    keyboard
end

