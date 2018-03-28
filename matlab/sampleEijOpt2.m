function [f, df] = sampleEijOpt2(x)
% optimization code
% no rotation case
    % param
    sample_grid = 1/6;
    sigma = 0.5;
    t = 0.9;
    gt = load('Mygt.mat');
    gt = gt.gt;
    gt_pt = gt.gt_pt;
    gt_pt_inv = gt.gt_pt_inv;
    ratio = max(min(1, size(gt_pt,1)/size(gt_pt_inv,1)*5),0.1);
    %ratio = 1;
    scale = x(1:3)';
    trans = x(4:6)';
    Rv = x(7:9)';
    % sampling 
    [sample_X,sample_Y,sample_Z] = meshgrid(0:sample_grid:1,0:sample_grid:1,0:sample_grid:1);
    sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];
    % affine
    theta = norm(Rv);
    Rv = Rv/(theta+eps);
    vx = getVX(Rv);
    Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
    cen_mean = mean(sample_pt);
    sample_pt_dst = bsxfun(@minus, sample_pt, cen_mean);
    sample_pt_dst = sample_pt_dst * Rrot;
    sample_pt_dst = bsxfun(@plus, sample_pt_dst, cen_mean);
    sample_pt_dst = bsxfun(@times, sample_pt, scale);
    sample_pt_dst = bsxfun(@plus, sample_pt_dst, trans);
    % loss
    pdis = pdist2(sample_pt_dst, gt_pt);
    pdis = pdis.*pdis;
    loss = exp(-pdis/(sigma*sigma));
    delta_V = scale(1)*scale(2)*scale(3)/size(sample_pt_dst,1); % scale-wise
    loss_tsdf = delta_V*min(t, loss);
    loss_t = -loss.*(loss<t);% truncated for gradient
    pdis_inv = pdist2(sample_pt_dst, gt_pt_inv);% inverse energy
    pdis_inv = pdis_inv.*pdis_inv;
    loss_inv = exp(-pdis_inv/(sigma*sigma));
    loss_tsdf_inv = delta_V*min(t, loss_inv);
    loss_t_inv = -loss_inv.*(loss_inv<t);
    
    f = -(sum(sum(loss_tsdf,2) - ratio*sum(loss_tsdf_inv,2))); % minimize
    
    % gradient
    loss_d = zeros(1,3);
    loss_r = zeros(1,3);
    loss_s = zeros(1,3);
    for i = 1:size(sample_pt_dst, 1)
        pi = sample_pt_dst(i,:); 
        co_term = sum(bsxfun(@minus, gt_pt, pi).*repmat(loss_t(i,:)',1,3));
        co_term =  delta_V*2*co_term/(sigma*sigma); % common term
        co_term_inv = sum(bsxfun(@minus, gt_pt_inv, pi).*repmat(loss_t_inv(i,:)',1,3));
        co_term_inv = delta_V*2*co_term_inv/(sigma*sigma);
        
        loss_d = loss_d + co_term; % translation
        loss_d = loss_d - ratio*co_term_inv;
        
        % rotation
        co_term = co_term.*sample_pt(i,:);
        co_term_inv = co_term_inv.*sample_pt(i,:);
        
%        Rv_d1_vx = getVX(cross(Rv', (eye-Rrot)*[1;0;0]));
%        Rv_d1 = co_term.*scale*((Rv(1)*vx+Rv_d1_vx)*Rrot/(theta+eps)/(theta+eps));
%        Rv_d2_vx = getVX(cross(Rv', (eye-Rrot)*[0;1;0]));
%        Rv_d2 = co_term.*scale*((Rv(2)*vx+Rv_d2_vx)*Rrot/(theta+eps)/(theta+eps));
%        Rv_d3_vx = getVX(cross(Rv', (eye-Rrot)*[0;0;1]));
%        Rv_d3 = co_term.*scale*((Rv(3)*vx+Rv_d3_vx)*Rrot/(theta+eps)/(theta+eps));
%        Rv_d1_inv = co_term_inv.*scale*((Rv(1)*vx+Rv_d1_vx)*Rrot/(theta+eps)/(theta+eps));
%        Rv_d2_inv = co_term_inv.*scale*((Rv(2)*vx+Rv_d2_vx)*Rrot/(theta+eps)/(theta+eps));
%        Rv_d3_inv = co_term_inv.*scale*((Rv(3)*vx+Rv_d3_vx)*Rrot/(theta+eps)/(theta+eps));
            
%        loss_r = loss_r + [Rv_d1(1), Rv_d2(2), Rv_d3(3)];% rotation
%        loss_r = loss_r - ratio*[Rv_d1_inv(1), Rv_d2_inv(2), Rv_d3_inv(3)];
        
        % scale
        Rs_dl = delta_V/scale(1)*sum(loss_t(i,:)) + co_term*Rrot;
        Rs_dw = delta_V/scale(2)*sum(loss_t(i,:)) + co_term*Rrot;
        Rs_dh = delta_V/scale(3)*sum(loss_t(i,:)) + co_term*Rrot;
%        Rs_dl = delta_V/scale(1)*sum(loss_t(i,:)) + co_term;
%        Rs_dw = delta_V/scale(2)*sum(loss_t(i,:)) + co_term;
%        Rs_dh = delta_V/scale(3)*sum(loss_t(i,:)) + co_term;
            
        Rs_dl_inv = delta_V/scale(1)*sum(loss_t_inv(i,:)) + co_term_inv*Rrot;
        Rs_dw_inv = delta_V/scale(2)*sum(loss_t_inv(i,:)) + co_term_inv*Rrot;
        Rs_dh_inv = delta_V/scale(3)*sum(loss_t_inv(i,:)) + co_term_inv*Rrot;
%        Rs_dl_inv = delta_V/scale(1)*sum(loss_t_inv(i,:)) + co_term_inv;
%        Rs_dw_inv = delta_V/scale(2)*sum(loss_t_inv(i,:)) + co_term_inv;
%        Rs_dh_inv = delta_V/scale(3)*sum(loss_t_inv(i,:)) + co_term_inv;
        loss_s = loss_s + [Rs_dl(1), Rs_dw(2), Rs_dh(3)];
        loss_s = loss_s - ratio*[Rs_dl_inv(1), Rs_dw_inv(2), Rs_dh_inv(3)];
    end
    
%    loss_r = loss_r *0.0001;
%    loss_s = loss_s * 0.1;
    df = [loss_s loss_d loss_r]';
%    df = [loss_s loss_d]';
    
end
