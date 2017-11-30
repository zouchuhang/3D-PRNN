-- adapted from https://github.com/jarmstrong2/handwritingnet/blob/master/yHatMat.lua

require 'nn'

local YHat, parent = torch.class('nn.YHat', 'nn.Module')

function YHat:updateOutput(input)
    local hat_e_t = input[{{},{1}}]
    local hat_pi_t = input[{{},{2,21}}]
    local hat_mu_1_t = input[{{},{22,41}}]
    local hat_mu_2_t = input[{{},{42,61}}]
    local hat_sigma_1_t = input[{{},{62,81}}]
    local hat_sigma_2_t = input[{{},{82,101}}]
    local hat_rho_t = input[{{},{102,121}}]

    self.e_t_act = self.e_t_act or nn.Sigmoid():cuda()
    self.pi_t_act = self.pi_t_act or nn.SoftMax():cuda()
    self.sigma_1_t_act = self.sigma_1_t_act or nn.Exp():cuda()
    self.sigma_2_t_act = self.sigma_2_t_act or nn.Exp():cuda()
    self.rho_t_act = self.rho_t_act or nn.Tanh():cuda()


    local e_t = self.e_t_act:forward(-hat_e_t)
    local pi_t = self.pi_t_act:forward(hat_pi_t)
    local mu_1_t = hat_mu_1_t:clone()
    local mu_2_t =  hat_mu_2_t:clone()
    local sigma_1_t = self.sigma_1_t_act:forward(hat_sigma_1_t)
    local sigma_2_t = self.sigma_2_t_act:forward(hat_sigma_2_t)
    local rho_t = self.rho_t_act:forward(hat_rho_t)
    
    local output = torch.cat(e_t:float(), pi_t:float(), 2)
    output = torch.cat(output, mu_1_t:float(), 2)
    output = torch.cat(output, mu_2_t:float(), 2)
    output = torch.cat(output, sigma_1_t:float(), 2)
    output = torch.cat(output, sigma_2_t:float(), 2)
    output = torch.cat(output, rho_t:float(), 2)
    self.output = output:cuda()
    
    return self.output
end

function YHat:updateGradInput(input, gradOutput)
    local hat_e_t = input[{{},{1}}]
    local hat_pi_t = input[{{},{2,21}}]
    local hat_mu_1_t = input[{{},{22,41}}]
    local hat_mu_2_t = input[{{},{42,61}}]
    local hat_sigma_1_t = input[{{},{62,81}}]
    local hat_sigma_2_t = input[{{},{82,101}}]
    local hat_rho_t = input[{{},{102,121}}]


    local d_e_t = gradOutput[{{},{1}}]
    local d_pi_t = gradOutput[{{},{2,21}}]
    local d_mu_1_t = gradOutput[{{},{22,41}}]
    local d_mu_2_t = gradOutput[{{},{42,61}}]
    local d_sigma_1_t = gradOutput[{{},{62,81}}]
    local d_sigma_2_t = gradOutput[{{},{82,101}}]
    local d_rho_t = gradOutput[{{},{102,121}}]

    local grad_e_t = d_e_t:clone() --self.e_t_act:backward(-hat_e_t, d_e_t)
    local grad_pi_t = d_pi_t:clone() --self.pi_t_act:backward(hat_pi_t, d_pi_t)
    local grad_mu_1_t = d_mu_1_t:clone()
    local grad_mu_2_t =  d_mu_2_t:clone()
    local grad_sigma_1_t = d_sigma_1_t:clone() --self.sigma_1_t_act:backward(hat_sigma_1_t, d_sigma_1_t)
    local grad_sigma_2_t = d_sigma_2_t:clone() --self.sigma_2_t_act:backward(hat_sigma_2_t, d_sigma_2_t)
    local grad_rho_t = d_rho_t:clone() --self.rho_t_act:backward(hat_rho_t, d_rho_t)
        
    local grad_input = torch.cat(grad_e_t:float(), grad_pi_t:float())
    grad_input = torch.cat(grad_input, grad_mu_1_t:float())
    grad_input = torch.cat(grad_input, grad_mu_2_t:float())
    grad_input = torch.cat(grad_input, grad_sigma_1_t:float())
    grad_input = torch.cat(grad_input, grad_sigma_2_t:float())
    grad_input = torch.cat(grad_input, grad_rho_t:float())
    
    self.gradInput = grad_input:cuda()
    
    return self.gradInput  
end

