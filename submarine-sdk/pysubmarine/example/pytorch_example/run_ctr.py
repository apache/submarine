from submarine.ml.pytorch.model.ctr import DeepFM

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", help="a JSON configuration file for FM", type=str)
    parser.add_argument("-task_type", default='train',
                        help="train or evaluate, by default is train")
    args = parser.parse_args()

    trainer = DeepFM(json_path=args.conf)

    if args.task_type == 'train':
        trainer.fit()
        print('[Train Done]')
    elif args.task_type == 'evaluate':
        score = trainer.evaluate()
        print(f'Eval score: {score}')
    elif args.task_type == 'predict':
        pred = trainer.predict()
        print('Predict:', pred)
    else:
        assert False, args.task_type
