"""Generate prepared TIFF/NPZ training samples from calibrated background and truth."""
import argparse
import json
from pathlib import Path
from stent_repro.preprocessing import prepare_samples

def main(argv=None):
    parser=argparse.ArgumentParser(description='CPU reference preprocessing; explicit calibrated parameters required.')
    parser.add_argument('--method',choices=['A','B'],required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--params',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    params=json.loads(Path(args.params).read_text(encoding='utf-8'))
    result=prepare_samples(args.manifest,args.output,method=args.method,profile=args.profile,params=params)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__ == '__main__': main()
