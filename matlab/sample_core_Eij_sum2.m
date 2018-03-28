% function to do sampling and compute loss
% loss is truncated for each Eij
% for optimization

% for rotation

function [loss, sample_pt_dst] = sample_core_Eij_sum2(sample_pt, trans, shape, scale, Rv, theta, gt)
    
    % affine transformation
    vx = getVX(Rv);% rotation
    %theta = norm(Rv);
    Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
    %disp(Rrot);
    sample_pt_dst = bsxfun(@plus, sample_pt, trans);
    cen_mean = mean(sample_pt_dst);
    sample_pt_dst = bsxfun(@minus, sample_pt_dst, cen_mean);
    sample_pt_dst = bsxfun(@times, sample_pt_dst, scale);
    sample_pt_dst = sample_pt_dst * Rrot;
    sample_pt_dst = bsxfun(@plus, sample_pt_dst, cen_mean);

    % voxel loss with gaussian smooth
    sigma = 2;
    t = 0.9;
    gt_pt = gt.gt_pt;
    gt_pt_inv = gt.gt_pt_inv;
    pdis = pdist2(sample_pt_dst, gt_pt);
    pdis = pdis.*pdis;
    loss = exp(-pdis/(sigma*sigma));
    delta_V = scale(1)*shape(1)*scale(2)*shape(2)*scale(3)*shape(3)/size(sample_pt_dst,1); % scale-wise
    loss_tsdf = delta_V*min(t, loss);
    % inverse energy
    pdis_inv = pdist2(sample_pt_dst, gt_pt_inv);
    pdis_inv = pdis_inv.*pdis_inv;
    loss_inv = exp(-pdis_inv/(sigma*sigma));
    loss_tsdf_inv = delta_V*min(t, loss_inv);
    %ratio = size(gt_pt,1)/size(gt_pt_inv,1);
    %ratio = 0.3;
    ratio = max(min(1, size(gt_pt,1)/size(gt_pt_inv,1)*5),0.1);
    
    % overlap loss

    loss = sum(sum(loss_tsdf,2) - ratio*sum(loss_tsdf_inv,2));
end