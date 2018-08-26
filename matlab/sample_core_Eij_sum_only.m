% function to do sampling and compute loss
% loss is truncated for each Eij
% for optimization

function [sample_pt_dst] = sample_core_Eij_sum_only(sample_pt, trans, scale, Rv, theta)
    % affine transformation
    vx = getVX(Rv);% rotation
    %theta = norm(Rv);
    Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
    %disp(Rrot);
    cen_mean = mean(sample_pt);
    sample_pt_dst = bsxfun(@times, sample_pt, scale);
    sample_pt_dst = bsxfun(@minus, sample_pt_dst, cen_mean);
    sample_pt_dst = sample_pt_dst * Rrot;
    sample_pt_dst = bsxfun(@plus, sample_pt_dst, trans);
    sample_pt_dst = bsxfun(@plus, sample_pt_dst, cen_mean);
end