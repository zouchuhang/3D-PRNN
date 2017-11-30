-- revised based on https://github.com/jarmstrong2/handwritingnet/blob/master/driver.lua

require 'cutorch'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Script for training sequence model.')

cmd:option('-inputSize' , 3, 'number of input dimension')
cmd:option('-hiddenSize' , 400, 'number of hidden units in lstms')
cmd:option('-lr' , 1e-4, 'learning rate')
cmd:option('-maxlen' , 100, 'max sequence length')
cmd:option('-batchSize' , 50, 'mini batch size')
cmd:option('-numPasses' , 1, 'number of passes')
cmd:option('-valData' , './data/prim_rnn_batch_nz_all_val.t7', 'filepath for validation data')
cmd:option('-trainData' , './data/prim_rnn_batch_nz_all_tr.t7', 'filepath for training data')

cmd:text()
opt = cmd:parse(arg)

dofile('model_3dp_depth.lua')
dofile('train_3dp_depth.lua')
