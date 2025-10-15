#!/usr/bin/env python3
import argparse, json, pathlib, sys
from transformers import AutoConfig
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-hf-dir', default='artifacts/stage01_hf')
    ap.add_argument('--onnx-dir', default='artifacts/stage03_onnx')
    ap.add_argument('--trust-remote-code', action='store_true')
    args = ap.parse_args()
    cfg = AutoConfig.from_pretrained(args.src_hf_dir, trust_remote_code=args.trust_remote_code)
    src_h = int(getattr(cfg,'num_attention_heads')); src_kv = int(getattr(cfg,'num_key_value_heads',src_h))
    print(f'Source GQA: heads={src_h}, kv_heads={src_kv}')
    cfgp = pathlib.Path(args.onnx_dir) / 'config.json'
    if cfgp.exists():
        j = json.loads(cfgp.read_text()); exp_h = int(j.get('num_attention_heads',0)); exp_kv = int(j.get('num_key_value_heads',exp_h))
        print(f'Export GQA: heads={exp_h}, kv_heads={exp_kv}')
        if (src_h,src_kv)==(exp_h,exp_kv): print('✓ GQA match'); sys.exit(0)
        else: print('✗ GQA mismatch'); sys.exit(3)
    else:
        print('No GQA metadata found in export; this can be normal.'); sys.exit(0)
if __name__=='__main__': main()
