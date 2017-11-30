-- revised based on https://github.com/jarmstrong2/handwritingnet/blob/master/getBatch.lua

require 'torch'
--require 'parsevocab'

sample_size = 3
function getMaxLen(newLen, remainingLen, count, data)
    maxLen = 0
    for i = count, remainingLen do
        inputLen = #(data[i].x_vals)
        if inputLen > maxLen then
            maxLen = inputLen
        end
    end
    for i = 1, newLen do
        inputLen = #(data[i].x_vals)
        if inputLen > maxLen then
            maxLen = inputLen
        end
    end
    return maxLen
end

function getLens(count, data)
    dataLen = #data
    if (count + (sample_size - 1)) > dataLen then
        newLen = (count + (sample_size - 1)) - dataLen
        remainingLen = (count + (sample_size - 1)) - newLen 
    else
        newLen = 0
        remainingLen = (count + (sample_size - 1))
    end
    return newLen, remainingLen
end

function getStrs(newLen, remainingLen, count, data)
    strs = {}
    for i = count, remainingLen do
        table.insert(strs, data[i].str)
    end
    for i = 1, newLen do
        table.insert(strs, data[i].str)
    end
    return strs
end

function getInputAndMaskMat(maxLen, newLen, remainingLen, count, data, voxdata)
    sampleCount = 1
    inputMat = torch.zeros(sample_size, 3, maxLen)
    rotMat = torch.zeros(sample_size, 2, maxLen)
    --sMat = torch.zeros(sample_size, 1, maxLen)
    ymaskMat = torch.zeros(sample_size, 121, maxLen)
    wmaskMat = torch.zeros(sample_size, 30, maxLen)
    cmaskMat = torch.zeros(sample_size, 1, maxLen)
    voxMat = torch.zeros(sample_size,1,64,64)
    elementCount = 0
    for i = count, remainingLen do
        for j = 1, #data[i].x_vals do
            inputMat[{{sampleCount}, {1}, {j}}] = data[i].x_vals[j]
            inputMat[{{sampleCount}, {2}, {j}}] = data[i].y_vals[j]
            inputMat[{{sampleCount}, {3}, {j}}] = data[i].e_vals[j]
            rotMat[{{sampleCount}, {1}, {j}}] = data[i].r_vals[j]
            rotMat[{{sampleCount}, {2}, {j}}] = data[i].rs_vals[j]
            --sMat[{{sampleCount}, {}, {j}}] = data[i].s_vals[j]
            ymaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            wmaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            cmaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            voxMat[{{sampleCount},{},{},{}}] = torch.reshape(voxdata[{{i},{},{}}],1,1,64,64)
            if j <= 1000 then
                elementCount = elementCount + 1
            end
        end
        sampleCount = sampleCount + 1
    end
    for i = 1, newLen do
        for j = 1, #data[i].x_vals do
            inputMat[{{sampleCount}, {1}, {j}}] = data[i].x_vals[j]
            inputMat[{{sampleCount}, {2}, {j}}] = data[i].y_vals[j]
            inputMat[{{sampleCount}, {3}, {j}}] = data[i].e_vals[j]
            rotMat[{{sampleCount}, {}, {j}}] = data[i].r_vals[j]
            --sMat[{{sampleCount}, {}, {j}}] = data[i].s_vals[j]
            ymaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            wmaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            cmaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            voxMat[{{sampleCount},{},{},{}}] = torch.reshape(voxdata[{{i},{},{}}],1,1,64,64)
            if j <= 1000 then
                elementCount = elementCount + 1
            end
        end
        sampleCount = sampleCount + 1
    end
    return inputMat, voxMat, rotMat, ymaskMat, wmaskMat, cmaskMat, elementCount
end

function getNewCount(newLen, remainingLen)
    if newLen == 0 then
        newCount = remainingLen + 1
    else
        newCount = newLen + 1
    end
    return newCount
end

function getBatch(count, data, sampsize, voxdata)
    sample_size = sampsize
    newLen, remainingLen = getLens(count, data)
    maxLen = getMaxLen(newLen, remainingLen, count, data)
    strs = getStrs(newLen, remainingLen, count, data)
    --cu = getOneHotStrs(strs)
    inputMat, voxMat, rotMat, ymaskMat, wmaskMat, cmaskMat, elementCount = getInputAndMaskMat(maxLen, newLen, remainingLen, count, data, voxdata)
    newCount = getNewCount(newLen, remainingLen)
    return maxLen, strs, inputMat, voxMat, rotMat, ymaskMat, wmaskMat, cmaskMat, elementCount, newCount
end
