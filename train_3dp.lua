-- revised from https://github.com/jarmstrong2/handwritingnet/blob/master/train_nngraph.lua

require 'getBatch'

sampleSize = opt.batchSize
numberOfPasses = opt.numPasses

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(sampleSize, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()
dfinalstate_h3_c = initstate_h1_c:clone()
dfinalstate_h3_h = initstate_h1_c:clone()

count = 1

batchCount = nil

function makecov(std, rho)
    covmat = torch.Tensor(2,2)
    covmat[{{1},{1}}] = torch.pow(std[{{1},{1}}], 2)
    covmat[{{1},{2}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{1}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{2}}] = torch.pow(std[{{1},{2}}], 2)
    return covmat
end

function getSample(sampleSize, yOutput)
    sampX = torch.zeros(sampleSize, 3)
    for i=1,sampleSize do
        currentY = yOutput[{{i},{}}]
        x_1, x_2, x_3 = _getSample(currentY)
        sampX[{{i},{1}}] = x_1
        sampX[{{i},{2}}] = x_2
        sampX[{{i},{3}}] = x_3
    end
    return sampX:cuda()
end

function _getSample(input)
    e_t = input[{{},{1}}]
    pi_t = input[{{},{2,21}}]
    mu_1_t = input[{{},{22,41}}]
    mu_2_t = input[{{},{42,61}}]
    sigma_1_t = input[{{},{62,81}}]
    sigma_2_t = input[{{},{82,101}}]
    rho_t = input[{{},{102,121}}]
    
    x_3 = torch.Tensor(1)
    x_3 = (x_3:bernoulli(e_t:squeeze())):squeeze()
    
    chosen_pi = torch.multinomial(pi_t:double(), 1):squeeze()

    curstd = torch.Tensor({{sigma_1_t[{{},{chosen_pi}}]:squeeze(), sigma_2_t[{{},{chosen_pi}}]:squeeze()}})
    curcor = torch.Tensor({{1, rho_t[{{},{chosen_pi}}]:squeeze()}})
    curcovmat = makecov(curstd, curcor)
    curmean = torch.Tensor({{mu_1_t[{{},{chosen_pi}}]:squeeze(), mu_2_t[{{},{chosen_pi}}]:squeeze()}})
    sample = distributions.mvn.rnd(curmean, curcovmat)
    x_1 = sample[1]
    x_2 = sample[2]
    return x_1, x_2, x_3
end

function getValLoss()
    local valnumberOfPasses = 15
    local valcount = 1
    local valsampleSize = opt.batchSize
    local loss = 0
    local loss_r = 0
    local loss_s = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, valnumberOfPasses do
	--print(i)
        --------------------- get mini-batch -----------------------
        
        maxLen, strs, inputMat, voxMat, rotMat, ymaskMat, wmaskMat, cmaskMat, elementCount, valcount = getBatch(valcount, valshapedata, valsampleSize, valData)
        ------------------------------------------------------------

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM

        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        local rot_pred = {} 
        -- forward
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{},{t+1}}]:squeeze(3)
            local rot_in = rotMat[{{},{},{t}}]:squeeze(3)
            local rot_target = rotMat[{{},{},{t+1}}]:squeeze(3)
            --local cond_context = voxMat
            -- model 
   
       output_y[t], lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], rot_pred[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), rot_in:cuda(), lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]}))

            -- criterion 
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss_r = loss_r + clones.rot_criterion[t]:forward(rot_pred[t], rot_target:cuda())
            loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss    
        end

        maxLen = nil
        strs = nil
        inputMat = nil
        voxMat = nil
	rotMat = nil 
        maskMat = nil
	rot_pred = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    return (loss+loss_r)/valnumberOfPasses, loss_r/valnumberOfPasses
end

function schedSampBool() 
    k = 0.9
    i = batchCount/80.0
    e_i = k^i
    -- if we get 1 then don't sample, if 0 then do sample
    randvar = torch.Tensor(1)
    result = randvar:bernoulli(e_i):squeeze()
    return result  
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    local loss = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, numberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, voxMat, rotMat, ymaskMat, wmaskMat, cmaskMat, elementCount, count = getBatch(count, shapedata, sampleSize, trainData)
        ------------------------------------------------------------
       -- print(count)
        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        
        
        local output_h1_w = {}
        local output_y = {}
        local rot_pred = {}
 
        -- forward
        
        --print('forward')
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{},{t+1}}]:squeeze(3)
            local rot_in = rotMat[{{},{},{t}}]:squeeze(3)
            local rot_target = rotMat[{{},{},{t+1}}]:squeeze(3)
            --local cond_context = voxMat
            
        output_y[t], lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], rot_pred[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), rot_in:cuda(),
                 lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]}))

            -- criterion 
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss_r = clones.rot_criterion[t]:forward(rot_pred[t], rot_target:cuda())
            loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss+loss_r 
        end
       
        elems = (elementCount - sampleSize) + elems
        
        -- backward
        
        --print('backward')
        
        local dlstm_c_h1 = dfinalstate_h1_c
        local dlstm_h_h1 = dfinalstate_h1_h
        local dlstm_c_h2 = dfinalstate_h2_c
        local dlstm_h_h2 = dfinalstate_h2_h
        local dlstm_c_h3 = dfinalstate_h3_c
        local dlstm_h_h3 = dfinalstate_h3_h
        
  
        for t = maxLen - 1, 1, -1 do
        
            local x_in = inputMat[{{},{},{t}}]:squeeze()
            local x_target = inputMat[{{},{},{t+1}}]:squeeze()
            local rot_in = rotMat[{{},{},{t}}]:squeeze()
            local rot_target = rotMat[{{},{},{t+1}}]:squeeze()
            --local cond_context = voxMat
            
            -- criterion
            local grad_crit = clones.criterion[t]:backward(output_y[t], x_target:cuda())
            local grad_crit_r = clones.rot_criterion[t]:backward(rot_pred[t], rot_target:cuda())
	    
            grad_crit:clamp(-100,100)            
	    grad_crit_r:clamp(-100,100)
            
           _x,_r, dlstm_c_h1, dlstm_h_h1,
            dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3 = unpack(clones.rnn_core[t]:backward({x_in:cuda(),rot_in:cuda(),
                 lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]},
                 {grad_crit, dlstm_c_h1, dlstm_h_h1,
                  dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3, grad_crit_r}))
            -- clip gradients
            dlstm_c_h1:clamp(-10,10)
            dlstm_h_h1:clamp(-10,10)
            dlstm_c_h2:clamp(-10,10)
            dlstm_h_h2:clamp(-10,10)
            dlstm_c_h3:clamp(-10,10)
            dlstm_h_h3:clamp(-10,10)

        end
    
        dh2_w = nil
        dh2_h1 = nil
        dh3_w = nil
        dh3_h2 = nil
        rot_pred = nil
        maxLen = nil
        strs = nil
        inputMat = nil
        voxMat = nil 
        rotMat = nil
	--cond_context = nil
        maskMat = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        dlstm_c_h1 = nil -- internal cell states of LSTM
        dlstm_h_h1 = nil -- internal cell states of LSTM
        dlstm_c_h2 = nil -- internal cell states of LSTM
        dlstm_h_h2 = nil -- internal cell states of LSTM
        dlstm_c_h3 = nil -- internal cell states of LSTM
        dlstm_h_h3 = nil -- internal cell states of LSTM
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    
    grad_params:div(numberOfPasses)
    
    -- clip gradient element-wise
    grad_params:clamp(-10, 10)
    
    
    return loss, grad_params
end

losses = {} 
vallosses = {}
vallosses_r = {}
vallosses_s = {}
local optim_state = {learningRate = opt.lr, alpha = 0.95, epsilon = 1e-6}
local iterations = 8000
local minValLoss = 1/0
for i = 1, iterations do
    batchCount = i

    local _, loss = optim.adam(feval, params, optim_state)

    print(string.format("update param, loss = %6.8f, gradnorm = %6.4e", loss[1], grad_params:clone():norm()))
    if i % 20 == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
        valLoss, valLoss2 = getValLoss()
        vallosses[#vallosses + 1] = valLoss - valLoss2
        vallosses_r[#vallosses_r + 1] = valLoss2
        
        print(string.format("validation loss = %6.8f", valLoss))
        print(string.format("validation prim loss = %6.8f", valLoss-valLoss2))
        print(string.format("validation rot loss = %6.8f", valLoss2))
  	
        if minValLoss > valLoss then
            minValLoss = valLoss
            params_save = params:clone()
            nn.utils.recursiveType(params_save,'torch.DoubleTensor')
            torch.save("./model/model_param_full.t7", params_save:double())
            model_save = model.rnn_core:clone()
            nn.utils.recursiveType(model_save,'torch.DoubleTensor')
            torch.save("./model/model_full.t7", model_save:double())
            print("------- Model Saved --------")
        end
        losses[#losses + 1] = loss[1]
        --torch.save("losses_full_trval.t7", losses)
        --torch.save("vallosses_full_trval.t7", vallosses)
        --torch.save("vallosses_r_full_trval.t7", vallosses_r)
    end
end
