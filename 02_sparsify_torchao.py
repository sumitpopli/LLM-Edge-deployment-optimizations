#!/usr/bin/env python3
import argparse, os, shutil, sys, pathlib

def copy_tree(src, dst):
    src = pathlib.Path(src); dst = pathlib.Path(dst)
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default=r'C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct')
    ap.add_argument('--out-dir', default=r'C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct_stage02_sparse')
    ap.add_argument('--density', type=float, default=0.5)   # recorded as a hint only
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--engine', choices=['torchao','none'], default='torchao')
    ap.add_argument('--fail-if-missing', action='store_true')
    args = ap.parse_args()

    if args.engine == 'none':
        copy_tree(args.in_dir, args.out_dir)
        print(f'✓ Stage02 (pass-through): {args.in_dir} -> {args.out_dir}')
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        try:
            from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
        except Exception as e:
            if args.fail_if_missing:
                print('[Stage02][error] torchao not available:', e); sys.exit(2)
            print('[Stage02][warn] torchao not available; pass-through.')
            copy_tree(args.in_dir, args.out_dir)
            print(f'✓ Stage02 (pass-through): {args.in_dir} -> {args.out_dir}')
            return

        # CUDA check — SemiSparseLinear acceleration is CUDA-only (Ampere+/cc>=8.0)
        if not torch.cuda.is_available():
            msg = '[Stage02][warn] CUDA not available; pass-through.'
            if args.fail_if_missing:
                print('[Stage02][error] CUDA not available and fail-if-missing set.'); sys.exit(2)
            print(msg); copy_tree(args.in_dir, args.out_dir)
            print(f'✓ Stage02 (pass-through): {args.in_dir} -> {args.out_dir}')
            return
        props = torch.cuda.get_device_properties(0)
        if props.major < 8:
            msg = f'[Stage02][warn] GPU cc {props.major}.{props.minor} < 8.0; pass-through.'
            if args.fail_if_missing:
                print('[Stage02][error]', msg); sys.exit(2)
            print(msg); copy_tree(args.in_dir, args.out_dir)
            print(f'✓ Stage02 (pass-through): {args.in_dir} -> {args.out_dir}')
            return

        print('[Stage02] Loading model…')
        _ = AutoConfig.from_pretrained(args.in_dir, trust_remote_code=args.trust_remote_code)
        tok = AutoTokenizer.from_pretrained(args.in_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
        m = AutoModelForCausalLM.from_pretrained(
            args.in_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code
        ).cuda().half().eval()

        # Build mapping {FQN_of_Linear: SemiSparseLinear}
        sparse_cfg = {
            name: SemiSparseLinear
            for name, mod in m.named_modules()
            if isinstance(mod, torch.nn.Linear)
        }
        if not sparse_cfg:
            print('[Stage02][warn] No nn.Linear modules found; pass-through.')
            copy_tree(args.in_dir, args.out_dir)
            print(f'✓ Stage02 (pass-through): {args.in_dir} -> {args.out_dir}')
            return

        print(f'[Stage02] Swapping {len(sparse_cfg)} Linear → SemiSparseLinear…')
        # NOTE: no `sparsity=` kwarg exists for this API (2:4 is fixed 50% pattern)
        swap_linear_with_semi_sparse_linear(m, sparse_cfg)

        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        # Record an informational hint only; swap does not persist a mask into weights.
        (pathlib.Path(args.out_dir) / 'sparsity_hint.json').write_text(
            '{"pattern":"2:4","density":%s}' % args.density
        )
        print(f'✓ Stage02: saved (swapped) model to {args.out_dir}')

    except Exception as e:
        if args.fail_if_missing:
            print('[Stage02][error] Sparsity failed and fail-if-missing set:', e); sys.exit(3)
        print('[Stage02][warn] Sparsity failed; pass-through:', e)
        copy_tree(args.in_dir, args.out_dir)
        print(f'✓ Stage02 (pass-through): {args.in_dir} -> {args.out_dir}')

if __name__ == '__main__':
    main()

