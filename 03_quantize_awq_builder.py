#!/usr/bin/env python3
import argparse, subprocess, sys, pathlib
def have(mod):
    try: __import__(mod); return True
    except Exception: return False
def sh(cmd):
    print(">>", " ".join(cmd)); subprocess.check_call(cmd)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default=r'C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct_stage02_sparse')
    ap.add_argument('--out-dir', default=r'C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct_stage03_onnx')
    ap.add_argument('--precision', choices=['int4','fp16'], default='int4')
    ap.add_argument('--ep', default='cpu')
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--fallback-optimum', action='store_true')
    args = ap.parse_args()
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if have('onnxruntime_genai'):
        cmd = [sys.executable,'-m','onnxruntime_genai.models.builder','-m',args.in_dir,'-o',args.out_dir,'-e',args.ep,'-p',args.precision]
        if args.trust_remote_code: cmd += ['--trust-remote-code']
        sh(cmd); print(f'✓ Stage03: {args.out_dir}'); return
    if not args.fallback_optimum:
        print('[Stage03][error] onnxruntime-genai not installed; use --fallback-optimum.'); sys.exit(2)
    base = ['optimum-cli','export','onnx','--model',args.in_dir,args.out_dir,'--task','text-generation']
    if args.precision=='fp16': base += ['--dtype','fp16']
    elif args.precision=='int4': base += ['--weight-format','int4']
    sh(base); print(f'✓ Stage03 (optimum fallback): {args.out_dir}')
if __name__ == '__main__':
    main()
