% visualize 

% res
res = load('../data/test_res_mn_pure.mat');
res = res.x;

for start_num = 1:size(res,1)
    
color = {'red','green','blue','magenta','cyan','yellow', 'black', 'white','black','white','black','white','black'};

figure(1); title('generation')% pred
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
        axis([0,30,0,30,0,30]); axis off
        %keyboard
end
        
keyboard
end

