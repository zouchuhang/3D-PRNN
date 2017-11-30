--require 'nn'
--local model_utils=require 'model_utils'

local VOXAEnet = {}

function VOXAEnet.voxae()

    local x = nn.Identity()()
    local conv1 = nn.SpatialConvolution(1,32,7,7,2,2)(x)
    local conv1_relu = nn.LeakyReLU(0.1,true)(conv1)

    local conv2 = nn.SpatialConvolution(32,64,5,5,2,2)(conv1_relu)
    local conv2_relu = nn.LeakyReLU(0.1,true)(conv2)
    local pool1 = nn.SpatialMaxPooling(2,2)(conv2_relu)

    local conv3 = nn.SpatialConvolution(64,128,3,3,2,2)(pool1)
    local conv3_relu = nn.LeakyReLU(0.1,true)(conv3)
    local pool2 = nn.SpatialMaxPooling(2,2)(conv3_relu)

    local view1 = nn.Reshape(128)(pool2)
    local linear1 = nn.Linear(128,64)(view1)
    local linear2 = nn.Linear(64,32)(linear1)

    return nn.gModule({x}, {linear2})
    
end

return VOXAEnet
