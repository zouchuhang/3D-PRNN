-- revised based on https://github.com/jarmstrong2/handwritingnet/blob/master/model_nngraph.lua

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'
require 'nngraph'
require 'optim'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'yHatMat2'
require 'mixtureCriterionMat'
local model_utils=require 'model_utils'
require 'cunn'
require 'distributions'
torch.manualSeed(123)
local matio = require 'matio'

-- get training dataset
dataFile = torch.DiskFile(opt.trainData, 'r')
inpdata = dataFile:readObject()
shapedata = {}
for i = 1, #inpdata, 5 do
    data_tmp = inpdata[i] 
    table.insert(shapedata, data_tmp)
end
dataSize = #shapedata
print(dataSize)
print('Uploaded training')

-- get validation dataset
valdataFile = torch.DiskFile(opt.valData, 'r')
valinpdata = valdataFile:readObject()
valshapedata = {}
for i = 1, #valinpdata, 5 do
    data_tmp = valinpdata[i]
    table.insert(valshapedata, data_tmp)
end
valdataSize = #valshapedata
print(valdataSize)
print('Uploaded validation')

-- get depth data
--print('vox data loading...')
trainData = matio.load('./data/depth_map/depth_mn_train_all_tr.mat')
trainData = trainData.depth_tile
ntrain = trainData:size(1)
--print(ntrain)
--print('vox validation')
valData = matio.load('./data/depth_map/depth_mn_train_all_val.mat')
valData = valData.depth_tile
nval = valData:size(1)

-- make model
model = {}

model.criterion = nn.MixtureCriterion():cuda()
model.criterion:setSizeAverage()
model.rot_criterion = nn.MSECriterion():cuda()

local input_xin = nn.Identity()()
local input_rin = nn.Identity()()
--local input_context = nn.Identity()()
local input_lstm_h1_h = nn.Identity()()
local input_lstm_h1_c = nn.Identity()()
local input_lstm_h2_h = nn.Identity()()
local input_lstm_h2_c = nn.Identity()()
local input_lstm_h3_h = nn.Identity()()
local input_lstm_h3_c = nn.Identity()()

local h1 = LSTMH1.lstm(opt.inputSize, opt.hiddenSize)({input_xin, input_rin, input_lstm_h1_c, input_lstm_h1_h})
local h1_c = nn.SelectTable(1)(h1)
local h1_h = nn.SelectTable(2)(h1)
local h2 = LSTMHN.lstm(opt.inputSize, opt.hiddenSize)({input_xin, input_rin, h1_h, input_lstm_h2_c, input_lstm_h2_h})
local h2_c = nn.SelectTable(1)(h2)
local h2_h = nn.SelectTable(2)(h2)
local h3 = LSTMHN.lstm(opt.inputSize, opt.hiddenSize)({input_xin, input_rin, h2_h, input_lstm_h3_c, input_lstm_h3_h})
local h3_c = nn.SelectTable(1)(h3)
local h3_h = nn.SelectTable(2)(h3)
local y = nn.YHat()(nn.Linear(opt.hiddenSize*3, 121)(nn.JoinTable(2)({h1_h, h2_h, h3_h})))
local rot_l1 = nn.Linear(opt.hiddenSize*3, 256)(nn.JoinTable(2)({h1_h, h2_h, h3_h}))
local rot_l2 = nn.Linear(256, 64)(rot_l1)
local rot_l2_relu = nn.ReLU(true)(rot_l2)
local rot_l3 = nn.Linear(64, 32)(rot_l2_relu)
local rot_l3_relu = nn.ReLU(true)(rot_l3)
local rot_res = nn.Linear(32, 1)(rot_l3_relu)
local rot_r2 = nn.Tanh()(rot_res)
local rot_res2 = nn.Linear(32, 1)(rot_l3)
local rot_rs2 = nn.Sigmoid()(rot_res2)
local rot_res3 = nn.JoinTable(2)({rot_r2, rot_rs2})

model.rnn_core = nn.gModule({input_xin, input_rin,
                             input_lstm_h1_c, input_lstm_h1_h,
                             input_lstm_h2_c, input_lstm_h2_h,
                             input_lstm_h3_c, input_lstm_h3_h},
                            {y, h1_c, h1_h, h2_c, h2_h,
                             h3_c, h3_h, rot_res3})

model.rnn_core:cuda()
params, grad_params = model.rnn_core:getParameters()


-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(1, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()

-- make a bunch of clones, AFTER flattening, as that reallocates memory
MAXLEN = opt.maxlen
clones = {} -- TODO: local
for name,mod in pairs(model) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times_fast(mod, MAXLEN-1, not mod.parameters)
end
print('start training')
