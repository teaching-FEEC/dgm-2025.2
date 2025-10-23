from CannyEdgeDetectorModel import DifferentiableCanny
from ProductionInferenceWrapper import RGBAWrapper
from SanityCheckModel import Identity3Channel
import torch
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample = DifferentiableCanny()#,dtype=torch.float16)
wrappedSample = RGBAWrapper(sample)
#wrappedSample.to(device)

dummy_input = torch.randint(0, 256, (1, 256, 256, 4), dtype=torch.uint8)#.to(device)

torch.onnx.export(
    wrappedSample,               # model being run
    dummy_input,                 # model input (or a tuple for multiple inputs)
    "wrapped_sample.onnx",       # where to save the model
    export_params=True,          # store trained parameter weights inside the model file
    opset_version=20,            # ONNX opset version
    do_constant_folding=True,    # optimize constants
    input_names=['input'],       # the model's input names
    output_names=['output'],     # the model's output names
    dynamic_axes={               # optional: allow variable batch size / H / W
        'input': {0: 'batch', 1: 'height', 2: 'width'},
        'output': {0: 'batch', 1: 'height', 2: 'width'}
    }
)