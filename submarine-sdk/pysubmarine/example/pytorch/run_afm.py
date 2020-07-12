# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from submarine.ml.pytorch.model.ctr import AttentionalFM

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-conf", help="a JSON configuration file for AttentionalFM", type=str)
    parser.add_argument("-task_type", default='train',
                        help="train or evaluate, by default is train")
    args = parser.parse_args()

    trainer = AttentionalFM(json_path=args.conf)

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
