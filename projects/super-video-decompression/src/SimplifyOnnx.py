import sys
import os
import onnxsim
import onnx


def simplify_onnx(input_path):
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_sim.onnx"

    print(f"Simplifying '{input_path}' → '{output_path}'...")
    model_simplified, check = onnxsim.simplify(input_path)
    
    if not check:
        print("Warning: Simplified model may be invalid.")
    
    #model_simplified.save(output_path)
    onnx.save(model_simplified, output_path)
    print(f"Simplified ONNX model saved to '{output_path}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simplify_onnx.py input_model.onnx")
        sys.exit(1)
    
    simplify_onnx(sys.argv[1])