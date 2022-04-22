#
# DirectML MNIST sample
# Based on the following model: https://github.com/onnx/models/blob/master/vision/classification/mnist/model/mnist-8.onnx
#

import pydirectml as dml
import numpy as np
import sys
import os

input_data = [10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0]
input_data_array = np.array(input_data, np.float32)

weight_data = [0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
               50.0, 50.0, 50.0]
weight_data_array = np.array(weight_data, np.float32)

bias_data = [6000, 7000, 8000, 9000]
bias_data_array = np.array(bias_data, np.float32)

input_bindings = []


def append_input_tensor(builder: dml.GraphBuilder, input_bindings: list, input_tensor: dml.TensorDesc, tensor_data_array):
    tensor = dml.input_tensor(builder, len(input_bindings), input_tensor)
    input_bindings.append(dml.Binding(tensor, tensor_data_array))
    return tensor


device = dml.Device(True, True)
builder = dml.GraphBuilder(device)

data_type = dml.TensorDataType.FLOAT32
input = dml.input_tensor(builder, 0, dml.TensorDesc(data_type, [1, 4, 2, 2]))
flags = dml.TensorFlags.OWNED_BY_DML

input_bindings.append(dml.Binding(input, input_data_array))

convolution_weight = append_input_tensor(builder, input_bindings, dml.TensorDesc(
    data_type, flags, [4, 1, 2, 2]), weight_data_array)
convolution_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(
    data_type, flags, [1, 4, 1, 1]), bias_data_array)
convolution = dml.convolution(input, convolution_weight, convolution_bias, strides=[
                              1, 1], start_padding=[0, 0], end_padding=[0, 0], group_count=4)

op = builder.build(dml.ExecutionFlags.NONE, [convolution])

# Compute the result
output_data = device.compute(op, input_bindings, [convolution])
output_tensor = np.array(output_data[0], np.float32)
print(output_tensor)
# The correct result should be [6010, 7046, 11000, 9000],
# microsoft.ai.directml.1.5.1 and microsoft.ai.directml.1.6.0 got the same correct result,
# but microsoft.ai.directml.1.7.0 and microsoft.ai.directml.1.8.0 actually got wrong result as [6010, 7000, 8000, 9000];
