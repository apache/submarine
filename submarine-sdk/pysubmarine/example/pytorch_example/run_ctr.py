from submarine.ml_pytorch.spec import Spec
from submarine.ml_pytorch.trainer import Trainer 

from argparse import ArgumentParser 

def parse_args(): 
    parser = ArgumentParser() 
    parser.add_argument('--conf', type=str, required=True) 
    parser.add_argument('--task_type', type=str, default='train') 
    args, _ = parser.parse_known_args() 
    return args 


if __name__ == '__main__': 
    args = parse_args() 

    trainer = Trainer(spec=Spec(json_path=args.conf)) 

    if args.task_type == 'train': 
        trainer.fit() 
        print('[Done]')
    elif args.task_type == 'evaluate': 
        score = trainer.evaluate() 
        print(f'Eval score: {score}')
    elif args.task_type == 'predict': 
        pred = trainer.predict() 
        print('Predict:', pred) 
    else: 
        assert False, args.task_type 
  