# LLM-Edge-deployment-optimizations
Optimizing hugging face LLMs to be optimized so that a single onnx model can be deployed on CPU, NPU, iGPU, GPU.
This requires 5 steps- 
1. Download the huggingface model from hf.
2. Sparsify the model to reduce the memory footprint using torchnao.
3. Quantize the model to int4 using AWQ and convert in to Onnx format.
4. Verify the model is GQA in hf and onnx format and all the optimizations and properties are still intact. 
5. Creating a fused model to run on NPU. Taking in to account the least powerful of all the silicon. 

## 01_download_model.py
Is a python script that downloads any huggingface model. 
### Pre-reqs:
1. pip install huggingface_hub
2. pip install hf_xet

<b>Commandline example:</b> python 01_download_model.py --repo-id 'Qwen/Qwen2.5-1.5B' --out-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B' --token xxxxx

## 02_sparsify_torchao.py
This script creates a sparse model where many weights are set to zero, making the model more efficient. 
### Pre-reqs:
1. pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126  <b>OR </b> pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision
(depending on your GPU card you might need a newer version.)
2. pip3 install transformers
3. pip3 install torchao

<b>Commandline Example:</b> python 02_sparsify_torchao.py --in-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B' --out-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B_stage02_sparse' --engine torchao

## 03_quantize_awq_builder.py
This script quantizes the model to int4 using AWQ scheme. Convertes the HF model to Onnx.
### Pre-reqs:
1. pip install onnxruntime_genai
2. pip install onnx_ir

<b>Commandline example:</b> python 03_quantize_awq_builder.py  --in-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B_stage02_sparse' --out-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B_stage03_onnx'

## 04_verify_gqa.py
This script verifies that the hf model and onnx model after conversion have the same GQA properties. 

<b>Commandline example:</b> python 04_verify_gqa.py --src-hf-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B' --onnx-dir 'C:/Users/sumit/Work/huggingface/Qwen2.5-1.5B_stage03_onnx'

## 05_fuse_script.txt
This script creates the final fused model for deployment. 

### Pre-reqs:
1. pip install olive-ai
<b>Commandline example:</b> olive run --config C:\Users\sumit\work\llama32_onnx_chain\olive_fuseonly.json



