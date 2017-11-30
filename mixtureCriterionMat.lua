-- revised based on: //github.com/jarmstrong2/handwritingnet/blob/master/mixtureCriterionMat.lua 

require 'nn'
local MixtureCriterion, parent = torch.class('nn.MixtureCriterion', 'nn.Criterion')

function MixtureCriterion:setmask(mask)
   self.mask = mask 
end

function MixtureCriterion:setSizeAverage()
   self.sizeAverage = true 
end

function MixtureCriterion:updateOutput(input, target)

    local x1 = target[{{},{1}}]
    local x2 = target[{{},{2}}]
    local x3 = target[{{},{3}}]
    
    local e_t = input[{{},{1}}]
    local pi_t = input[{{},{2,21}}]
    local mu_1_t = input[{{},{22,41}}]
    local mu_2_t = input[{{},{42,61}}]
    local sigma_1_t = input[{{},{62,81}}]
    local sigma_2_t = input[{{},{82,101}}]
    local rho_t = input[{{},{102,121}}]
    
    local sampleSize = (#input)[1]
    
    local inv_sigma1 = torch.pow(sigma_1_t + (10^-15), -1)
    local inv_sigma2 = torch.pow(sigma_2_t + (10^-15), -1)
    
    local mixdist1 = torch.cmul(inv_sigma1, inv_sigma2)
    mixdist1:cmul(torch.pow((-(torch.pow(rho_t, 2)) + 1 + (10^-15)), -0.5))
    mixdist1:mul(1/(2*math.pi))
    
    local mu_1_x_1 = (mu_1_t:clone()):mul(-1)
    local mu_2_x_2 = (mu_2_t:clone()):mul(-1)
    
    local x1_val = x1:expand(sampleSize, 20)
    local x2_val = x2:expand(sampleSize, 20) 
    mu_1_x_1:add(x1_val)
    mu_2_x_2:add(x2_val) 
    
    local mixdist2_z_1 = torch.cmul(torch.pow(inv_sigma1, 2), torch.pow(mu_1_x_1,2))  
    local mixdist2_z_2 = torch.cmul(torch.pow(inv_sigma2, 2), torch.pow(mu_2_x_2,2)) 
    local mixdist2_z_3 = torch.cmul(inv_sigma1, inv_sigma2)
    
    mixdist2_z_3:cmul(mu_1_x_1)
    mixdist2_z_3:cmul(mu_2_x_2)
    
    mixdist2_z_3:cmul(torch.mul(rho_t, 2))
    local z = mixdist2_z_1 + mixdist2_z_2 - mixdist2_z_3
    local mixdist2 = z:clone()
    mixdist2:mul(-1)
    mixdist2:cmul(torch.pow((-(torch.pow(rho_t, 2)) + 1+ (10^-15)):mul(2), -1))
    --print('log mixdist2', mixdist2:sum())
    mixdist2:exp()
    local mixdist = torch.cmul(mixdist1, mixdist2)
    mixdist:cmul(pi_t)
    
    local mixdist_sum = torch.sum(mixdist, 2)
    mixdist_sum:add(10^-15)    
 
    local log_mixdist_sum = torch.log(mixdist_sum)
    
    local log_e_t = e_t:clone()
    
    local eq1 = torch.eq(x3, torch.ones(sampleSize, 1):cuda())
    eq1 = eq1:cuda()
    eq1:cmul(torch.log(e_t+(10^-15)))
    local neq1 = torch.ne(x3, torch.ones(sampleSize, 1):cuda())
    neq1 = neq1:cuda()
    neq1:cmul(torch.log(-e_t + 1+(10^-15)))
    local log_e_t = eq1 + neq1
    
    local result = log_mixdist_sum + log_e_t
    result:mul(-1)
    result:cmul(self.mask)
    result = result:sum()
    if self.sizeAverage then
        result = result/target:size(1)
    end
    
    return result
end

function MixtureCriterion:updateGradInput(input, target)
    local x1 = target[{{},{1}}]
    local x2 = target[{{},{2}}]
    local x3 = target[{{},{3}}]
    
    local e_t = input[{{},{1}}]
    local pi_t = input[{{},{2,21}}]
    local mu_1_t = input[{{},{22,41}}]
    local mu_2_t = input[{{},{42,61}}]
    local sigma_1_t = input[{{},{62,81}}]
    local sigma_2_t = input[{{},{82,101}}]
    local rho_t = input[{{},{102,121}}]

    local sampleSize = (#input)[1]
    
    --responsibilities will separate calculation into gamma1 and gamma2
    
    local inv_sigma1 = torch.pow(sigma_1_t + (10^-15), -1)
    local inv_sigma2 = torch.pow(sigma_2_t + (10^-15), -1)
    
    local gamma1 = torch.cmul(inv_sigma1, inv_sigma2)
    gamma1:cmul(torch.pow((-(torch.pow(rho_t, 2)) + 1 + (10^-15)), -0.5))
    gamma1:mul(1/(2*math.pi))
    
    local mu_1_x_1 = (mu_1_t:clone()):mul(-1)
    local mu_2_x_2 = (mu_2_t:clone()):mul(-1)
    
    local x1_val = x1:expand(sampleSize, 20)
    local x2_val = x2:expand(sampleSize, 20) 
    mu_1_x_1:add(x1_val)
    mu_2_x_2:add(x2_val) 
    
    local gamma2_z_1 = torch.cmul(torch.pow(inv_sigma1, 2), torch.pow(mu_1_x_1,2))  
    local gamma2_z_2 = torch.cmul(torch.pow(inv_sigma2, 2), torch.pow(mu_2_x_2,2))
    
    local gamma2_z_3 = torch.cmul(inv_sigma1, inv_sigma2)
    gamma2_z_3:cmul(mu_1_x_1)
    gamma2_z_3:cmul(mu_2_x_2)
    
    gamma2_z_3:cmul(torch.mul(rho_t, 2))
    local z = gamma2_z_1 + gamma2_z_2 - gamma2_z_3
    
    local gamma2 = z:clone()
    gamma2:mul(-1)
    gamma2:cmul(torch.pow((-(torch.pow(rho_t, 2)) + 1 + (10^-15)):mul(2), -1))
    gamma2:exp()
    local gamma = torch.cmul(gamma1, gamma2)
    gamma:cmul(pi_t)
    local gamma_sum = torch.sum(gamma, 2)
    gamma_sum:add(10^-15)

    local gamma_sum_val = gamma_sum:expand(sampleSize, 20)
    gamma:cmul(torch.pow(gamma_sum_val, -1))

    local dl_hat_e_t = e_t:clone()
    dl_hat_e_t:mul(-1)
    
    dl_hat_e_t:add(x3)
    
    local dl_hat_pi_t = pi_t - gamma
    
    local c = torch.pow((-torch.pow(rho_t, 2)):add(1 + 10^-15), -1)
    
    local c_sigma1 = torch.cmul(c, inv_sigma1)
    local x1_mu1_sigma1 = torch.cmul(mu_1_x_1, inv_sigma1)
    local cor_x_2_mu2_sigma2 = torch.cmul(mu_2_x_2, rho_t)
    cor_x_2_mu2_sigma2:cmul(inv_sigma2)
    local dl_hat_mu_1_t = torch.cmul(x1_mu1_sigma1 - cor_x_2_mu2_sigma2, c_sigma1)
    dl_hat_mu_1_t:cmul(-gamma)
    
    local c_sigma2 = torch.cmul(c, inv_sigma2)
    local x2_mu2_sigma2 = torch.cmul(mu_2_x_2, inv_sigma2)
    local cor_x_1_mu1_sigma1 = torch.cmul(mu_1_x_1, rho_t)
    cor_x_1_mu1_sigma1:cmul(inv_sigma1)
    local dl_hat_mu_2_t = torch.cmul(x2_mu2_sigma2 - cor_x_1_mu1_sigma1, c_sigma2)
    dl_hat_mu_2_t:cmul(-gamma)
    
    local dl_hat_sigma_1_t = torch.cmul(c, mu_1_x_1)
    dl_hat_sigma_1_t:cmul(inv_sigma1)
    dl_hat_sigma_1_t:cmul(x1_mu1_sigma1 - cor_x_2_mu2_sigma2)
    dl_hat_sigma_1_t:add(-1)
    dl_hat_sigma_1_t:cmul(-gamma)
    
    local dl_hat_sigma_2_t = torch.cmul(c, mu_2_x_2)
    dl_hat_sigma_2_t:cmul(inv_sigma2)
    dl_hat_sigma_2_t:cmul(x2_mu2_sigma2 - cor_x_1_mu1_sigma1)
    dl_hat_sigma_2_t:add(-1)
    dl_hat_sigma_2_t:cmul(-gamma)
    
    local dl_hat_rho_t = torch.cmul(mu_1_x_1, mu_2_x_2)
    dl_hat_rho_t:cmul(inv_sigma1)
    dl_hat_rho_t:cmul(inv_sigma2)
    local cz = torch.cmul(c, z)
    local rho_cz = torch.cmul(rho_t, (-cz) + 1)
    local dl_hat_rho_t = dl_hat_rho_t + rho_cz
    dl_hat_rho_t:cmul(-gamma)

    local grad_input = torch.cat(dl_hat_e_t:float(), dl_hat_pi_t:float())
    grad_input = torch.cat(grad_input, dl_hat_mu_1_t:float())
    grad_input = torch.cat(grad_input, dl_hat_mu_2_t:float())
    grad_input = torch.cat(grad_input, dl_hat_sigma_1_t:float())
    grad_input = torch.cat(grad_input, dl_hat_sigma_2_t:float())
    grad_input = torch.cat(grad_input, dl_hat_rho_t:float())
    
    self.gradInput = grad_input:cuda()
    self.gradInput:cmul(self.mask:reshape(self.mask:size(1),1):expand(self.gradInput:size()))
    
    if self.sizeAverage then
        self.gradInput:div(self.gradInput:size(1))
    end
    return self.gradInput
end
