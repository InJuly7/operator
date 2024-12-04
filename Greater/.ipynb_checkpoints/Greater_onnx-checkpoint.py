# import onnx
# from onnx import helper
# from onnx import TensorProto

# # Step 1: Create the model
# # Create two input tensors
# input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 10])
# input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 10])

# # Create the Greater operator node
# greater_node = helper.make_node(
#     'Greater',  # Operation type
#     ['input1', 'input2'],  # Inputs
#     ['output'],  # Output
# )

# # Create a graph that contains the Greater node
# graph = helper.make_graph(
#     [greater_node],  # Nodes in the graph
#     'GreaterGraph',  # Name of the graph
#     [input1, input2],  # Inputs to the graph
#     [helper.make_tensor_value_info('output', TensorProto.BOOL, [1, 10])]  # Output tensor
# )

# # Create the ONNX model with opset_version 17
# model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[helper.make_opsetid('onnx', 17)])

# # Save the model to a file
# onnx.save(model, 'greater_model_opset17.onnx')

import onnxruntime as ort
import numpy as np

# Step 2: Run inference using onnxruntime
# Load the model
session = ort.InferenceSession('greater_model_opset17.onnx')

# Create input data (numpy arrays)
input_data1 = np.random.rand(1, 10).astype(np.float32)  # Ensure it has the shape [1, 10]
input_data2 = np.random.rand(1, 10).astype(np.float32)  # Ensure it has the shape [1, 10]

# 打印向量
print("input_data1:", input_data1)
print("input_data2:", input_data2)

# Set the inputs for the model
inputs = {
    'input1': input_data1,
    'input2': input_data2
}

# Run the model (perform inference)
outputs = session.run(None, inputs)

# The output will be a numpy array with the result of Greater operation
print("Output:", outputs[0])  # Expecting a boolean result
