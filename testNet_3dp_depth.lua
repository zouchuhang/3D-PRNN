-- revised from https://github.com/jarmstrong2/handwritingnet/blob/master/testNet.lua

require 'nn'
require 'torch'
torch.setdefaulttensortype('torch.DoubleTensor')
require 'nngraph'
require 'yHatMat_nocuda'
require 'distributions'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
local VOXnet = require 'depth_ae'
require 'getBatch'
matio = require 'matio'


-- load model
model = torch.load('./model/model_mn_depth_tr.t7')

-- nearest neighbor retrieved feature to initialize the first axes of the first primitive
test_samp = matio.load('./data/sample_generation/test_NNfeat_mn_chair.mat')

-- data
-- get training dataset
dataFile = torch.DiskFile('./data/prim_rnn_batch_nz_all_tr.t7', 'r')
shapedata = dataFile:readObject()
dataSize = #shapedata
print('vox data loading...')

trainData = matio.load('./data/depth_map/depth_mn_test_chair_ts.mat')
trainData = trainData.depth_tile
ntrain = trainData:size(1)

--iteration = 100
rs_res = torch.zeros(4*test_samp.test_ret_num:size(1), 100)

for samp_num = 1, test_samp.test_ret_num:size(1) do
    print(samp_num)
    -- get test sample
    maxLen, strs, inputMat, voxMat, rotMat, ymaskMat, wmaskMat, cmaskMat, elementCount, count = getBatch(samp_num, shapedata, 1, trainData)
    
x_val = {[1]=test_samp.test_sample[{{1},{samp_num}}][1][1]}
y_val = {[1]=test_samp.test_sample[{{2},{samp_num}}][1][1]}
e_val = {[1]=0}
r_val = {[1]=test_samp.test_sample[{{3},{samp_num}}][1][1]}
s_val = {[1]=test_samp.test_sample[{{4},{samp_num}}][1][1]}    

lstm_c_h1 = torch.zeros(1, 400)
lstm_h_h1 = torch.zeros(1, 400)
lstm_c_h2 = torch.zeros(1, 400)
lstm_h_h2 = torch.zeros(1, 400)
lstm_c_h3 = torch.zeros(1, 400)
lstm_h_h3 = torch.zeros(1, 400)

function makecov(std, rho)
    covmat = torch.Tensor(2,2)
    covmat[{{1},{1}}] = torch.pow(std[{{1},{1}}], 2)
    covmat[{{1},{2}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{1}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{2}}] = torch.pow(std[{{1},{2}}], 2)
    return covmat
end


function getX(input)
    
    e_t = input[{{},{1}}]
    pi_t = input[{{},{2,21}}]
    mu_1_t = input[{{},{22,41}}]
    mu_2_t = input[{{},{42,61}}]
    sigma_1_t = input[{{},{62,81}}]
    sigma_2_t = input[{{},{82,101}}]
    rho_t = input[{{},{102,121}}]
    
    x_3 = torch.Tensor(1)
    x_3 = (x_3:bernoulli(e_t:squeeze())):squeeze()

    choice = {}
    
    for k=1,10 do
       table.insert(choice, distributions.cat.rnd(pi_t:squeeze(1)):squeeze()) 
    end
    --chosen_pi = torch.multinomial(pi_t, 1):squeeze()
    --print(chosen_pi)
    _,chosen_pi = torch.max(pi_t,2)
    chosen_pi = chosen_pi:squeeze()
    --print(chosen_pi)

    randChoice = torch.random(10)
    
    max = 0
    for i=1,20 do
        cur = pi_t[{{},{i}}]:squeeze()
        if cur > max then
           max = cur 
            index = i
        end
    end
    
    curstd = torch.Tensor({{sigma_1_t[{{},{chosen_pi}}]:squeeze(), sigma_2_t[{{},{chosen_pi}}]:squeeze()}})
    curcor = torch.Tensor({{1, rho_t[{{},{chosen_pi}}]:squeeze()}})
    curcovmat = makecov(curstd, curcor)
    curmean = torch.Tensor({{mu_1_t[{{},{chosen_pi}}]:squeeze(), mu_2_t[{{},{chosen_pi}}]:squeeze()}})
    sample = distributions.mvn.rnd(curmean, curcovmat)
    --x_1 = sample[1]
    --x_2 = sample[2]
    --print(e_t)
    x_1 = curmean[1][1]
    x_2 = curmean[1][2]

    table.insert(x_val, x_1)
    table.insert(y_val, x_2)
    table.insert(e_val, x_3)
end

--priming the network
--kappaprev = torch.zeros(1, 10)
--w = torch.zeros((cu[{{},{1},{}}]:squeeze(2)):size())

for t=1, 40 do
    x_in = torch.Tensor({{x_val[t], y_val[t], e_val[t]}})
    r_in = torch.Tensor({{r_val[t],s_val[t]}})
    --s_in = torch.Tensor({{s_val[t]}})
    cond_context = voxMat
-- model 
        
        output_y, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3, w_vector, rot_res
    = unpack(model:forward({x_in, r_in,cond_context, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3})) 
        
    getX(output_y)
    table.insert(r_val, rot_res[1][1])
    table.insert(s_val, rot_res[1][2])
end

mean_x = test_samp.mean_x[1][1]
mean_y = test_samp.mean_y[1][1]
mean_r = test_samp.mean_r[1][1]
std_x = test_samp.std_x[1][1]
std_y = test_samp.std_y[1][1]
std_r = test_samp.std_r[1][1]

for t=1,40 do
    x_val[t] = (x_val[t]*std_x) + mean_x
    y_val[t] = (y_val[t]*std_y) + mean_y
    r_val[t] = (r_val[t]*std_r) + mean_r
end

rs = {}
new = true
count = 0
i = 0
oldx = 0
oldy = 0
for j=1,40 do
    i = i + 1
    if new then
        count = count + 1
        table.insert(rs, torch.zeros(4, 40))
        i = 1
        new = false
    end
    
    if e_val[j] == 1 or j == 40 then
    --if j == 40 then
        new = true
        rs[count] = rs[count][{{},{1,i}}]
    end
    
    newx = x_val[j]
    newy = y_val[j]
    newr = r_val[j]
    --newx = oldx + x_val[j]
    --newy = oldy - y_val[j] 
    --newy = oldy + y_val[j]
    --oldx = newx
    --oldy = newy
    rs[count][{{1},{i}}] = newx
    rs[count][{{2},{i}}] = newy
    rs[count][{{3},{i}}] = newr
    rs[count][{{4},{i}}] = s_val[j]
    --rs[count][{{5},{i}}] = e_val[j]
end

  
rs_res[{{(samp_num-1)*4+1, samp_num*4},{1,rs[1]:size(2)}}] = rs[1]
    if samp_num % 50 == 0 then
        matio.save('./result/test_res_mn_chair.mat' , rs_res)
    end

end

matio.save('./result/test_res_mn_chair.mat' , rs_res)
