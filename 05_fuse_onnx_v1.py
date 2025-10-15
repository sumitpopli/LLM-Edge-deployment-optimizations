#!/usr/bin/env python3
# olive_fuse_only.py
# Purpose: Use Microsoft Olive to apply ONLY transformer graph fusions (optimization_level=2) to an ONNX model.
# Requires: pip install olive-ai (CLI "olive" must be on PATH)
#
# Example (Llama 3.2 1B):
#   python olive_fuse_only.py ^
#     --model "C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct_stage03_onnx\model.onnx" ^
#     --out_dir "C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct_stage03_onnx_olive_fused" ^
#     --model-type llama --num-heads 16 --hidden-size 2048 --opt-level 2 --verbose
#
# Example (Qwen2.5-1.5B):
#   python olive_fuse_only.py ^
#     --model "C:\Users\sumit\Work\huggingface\Qwen2.5-1.5B_stage03_onnx\model.onnx" ^
#     --out_dir "C:\Users\sumit\Work\huggingface\Qwen2.5-1.5B_stage03_onnx_olive_fused" ^
#     --model-type qwen --num-heads 12 --hidden-size 1536 --opt-level 2

import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile

def _resolve_onnx(path_str: str) -> str:
    p = pathlib.Path(path_str)
    if p.is_dir():
        cands = sorted(p.glob("*.onnx"))
        if not cands:
            raise FileNotFoundError(f"No .onnx in directory: {p}")
        return str(cands[0])
    if not p.exists():
        raise FileNotFoundError(f"ONNX not found: {p}")
    return str(p)

def _guess_model_type(path_str: str):
    s = path_str.lower()
    if "llama" in s: return "llama"
    if "qwen"  in s: return "qwen"
    if "gpt"   in s: return "gpt2"
    if any(k in s for k in ["bert","roberta","deberta"]): return "bert"
    return None

def main():
    ap = argparse.ArgumentParser(description="Run ONLY transformer fusions with Olive (OrtTransformersOptimization).")
    ap.add_argument("--model", required=True, help="Path to model.onnx or a dir containing it.")
    ap.add_argument("--out_dir", required=True, help="Where Olive will write the fused model.")
    ap.add_argument("--model-type", default="auto", choices=["auto","llama","qwen","gpt2","bert"])
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--hidden-size", type=int, default=None)
    ap.add_argument("--opt-level", type=int, default=2, choices=[0,1,2,99], help="ORT transformers optimization level.")
    ap.add_argument("--disable-attention-fusion", action="store_true")
    ap.add_argument("--disable-gelu-fusion", action="store_true")
    ap.add_argument("--disable-layernorm-fusion", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_path = _resolve_onnx(args.model)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "olive_cache"
    cache_dir.mkdir(exist_ok=True)

    # Check that the Olive CLI is available
    if shutil.which("olive") is None:
        sys.stderr.write(
            "[error] 'olive' CLI not found. Install with: pip install olive-ai\n"
            "        Then ensure your environment PATH includes the Scripts directory (Windows) or bin (Unix).\n"
        )
        sys.exit(1)

    # Decide model_type and default dims if missing
    model_type = args.model_type
    if model_type == "auto":
        model_type = _guess_model_type(model_path) or "bert"
        if args.verbose:
            print(f"[info] Auto-detected model_type={model_type}")

    num_heads = args.num_heads
    hidden_size = args.hidden_size
    if num_heads is None or hidden_size is None:
        if args.verbose:
            print("[info] num_heads/hidden_size not fully specified; applying family defaults.")
        if model_type == "llama":
            num_heads   = num_heads   or 32    # Llama-2 7B default
            hidden_size = hidden_size or 4096
        elif model_type == "qwen":
            num_heads   = num_heads   or 12    # Qwen2.5-1.5B default
            hidden_size = hidden_size or 1536
        elif model_type in ("gpt2","bert"):
            num_heads   = num_heads   or 12
            hidden_size = hidden_size or 768

    # Build minimal Olive config: one pass -> OrtTransformersOptimization
    # Note: provider/system selection is irrelevant for offline fusions,
    # but Olive wants a system block; we'll keep CPUExecutionProvider.
    conf = {
        "engine": {
            "output_dir": str(out_dir),
            "cache_dir": str(cache_dir)
        },
        "systems": {
            "local": {
                "type": "LocalSystem",
                "inference_settings": {
                    "provider": "CPUExecutionProvider"
                }
            }
        },
        "passes": {
            "ort_transformers": {
                "type": "OrtTransformersOptimization",
                "config": {
                    "model_type": model_type,
                    "num_heads": num_heads,
                    "hidden_size": hidden_size,
                    # Different Olive builds may use 'optimization_level' or 'opt_level'.
                    # We set both; extra keys are ignored by tolerant parsers.
                    "optimization_level": args.opt_level,
                    "opt_level": args.opt_level,
                    # Map per-fusion toggles (default is enabled; we let you disable)
                    "enable_attention_fusion": not args.disable_attention_fusion,
                    "disable_gelu_fusion": args.disable_gelu_fusion,
                    "disable_layer_norm_fusion": args.disable_layernorm_fusion,
                    # Ensure we do not alter precision:
                    "forced_fp16": False
                }
            }
        },
        "inputs": {
            "model": model_path
        },
        "package_config": {
            "type": "ONNXModel"
        },
        # no evaluators; we’re only transforming the graph
    }

    # Write a throwaway config and run Olive
    with tempfile.TemporaryDirectory() as td:
        cfg_path = pathlib.Path(td) / "olive_fuse_only.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=2)
        if args.verbose:
            print(f"[debug] Olive config written to: {cfg_path}")
            print(json.dumps(conf, indent=2))

        cmd = ["olive", "run", "--config", str(cfg_path)]
        if args.verbose:
            print("[cmd]", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=not args.verbose, text=True)

        if res.returncode != 0:
            # Print captured output for debugging
            if not args.verbose:
                sys.stderr.write(res.stdout or "")
                sys.stderr.write(res.stderr or "")
            sys.stderr.write("\n[error] Olive run failed. See logs above.\n")
            sys.exit(res.returncode)

    print(f"[done] Fused model written under: {out_dir}")
    print("[hint] Use the SAME fused ONNX for CPU (CPUExecutionProvider), DML (DmlExecutionProvider), "
          "Intel NPU (OpenVINOExecutionProvider device_type='NPU'), or AMD NPU (vendor EP).")

if __name__ == "__main__":
    main()
