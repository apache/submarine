from tqdm import tqdm 

import math 
from collections import defaultdict 
from argparse import ArgumentParser 

NUM_FIELDS = 39 
FIELDS_I = list(range(13))
FIELDS_C = list(range(13, 39))

def parse_args(): 
    parser = ArgumentParser() 
    parser.add_argument('--input_path', type=str, required=True) 
    parser.add_argument('--output_path', type=str, required=True) 
    parser.add_argument('--threshold', type=int, default=10) 
    args = parser.parse_args() 
    return args 

def discretize_I_feature(value: str): 
    if value == '': 
        return 'NULL' 
    value = int(value) 
    if value > 2: 
        return str(int(math.log(value)**2)) 
    else: 
        return str(value - 2) 


if __name__ == '__main__': 
    print('[parse args]')
    args = parse_args()
    print(args) 
    
    print('[init field feature counter]')
    feature_counts = [defaultdict(int) for _ in range(NUM_FIELDS)] 

    print('[count field features]')
    with open(args.input_path, mode='r') as f: 
        for line in tqdm(f, desc='build feature mapping'): 
            label, *values = line.rstrip('\n').split('\t') 
            for field_id in FIELDS_I: 
                feature_counts[field_id][discretize_I_feature(values[field_id])] += 1 
            for field_id in FIELDS_C: 
                feature_counts[field_id][values[field_id]] += 1 
    
    print('[build feature mapping]') 
    feature_mapping = [
            {
                feat: feat_id for feat_id, feat in enumerate(
                    feat for feat, count in field.items() if count >= args.threshold
                )
            } for field in feature_counts 
    ] 
    feature_default = [len(m) for m in feature_mapping]
    
    print('[generate libsvm]') 
    with open(args.input_path, mode='r') as fin, open(args.output_path, mode='w') as fout: 
        for inline in tqdm(fin, desc='generate libsvm'): 
            label, *values = inline.rstrip('\n').split('\t') 
            outline = ''.join([
                label,
                *(f' {field_id}:{feature_mapping[field_id].get(discretize_I_feature(values[field_id]), feature_default[field_id])}' for field_id in FIELDS_I), 
                *(f' {field_id}:{feature_mapping[field_id].get(values[field_id], feature_default[field_id])}' for field_id in FIELDS_C)
            ])
            fout.write(f'{outline}\n')

    print('[field dims]') 
    print([len(field)+1 for field in feature_mapping])
    print('[done]')




