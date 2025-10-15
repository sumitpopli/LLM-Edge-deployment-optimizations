#!/usr/bin/env python3
import argparse, os
from huggingface_hub import snapshot_download
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', default='meta-llama/Llama-3.2-1B-Instruct')
    ap.add_argument('--out-dir', default=r'C:\Users\sumit\Work\huggingface\Llama-3.2-1B-Instruct')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--token', default="xxx", help='HF token or set HF_TOKEN env var')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    token = args.token or os.environ.get('HF_TOKEN')
    path = snapshot_download(repo_id=args.repo_id, local_dir=args.out_dir,
                             local_dir_use_symlinks=False, resume_download=True,
                             revision=args.revision, token=token)
    print(f'✓ Stage01: downloaded to {path}')
if __name__ == '__main__':
    main()
