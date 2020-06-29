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

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from submarine.utils.fileio import read_file


class LIBSVMDataset(Dataset):

    def __init__(self, path):
        self.data, self.label = self.preprocess_data(read_file(path))

    def __getitem__(self, idx):
        return self.data.iloc[idx], self.label.iloc[idx]

    def __len__(self):
        return len(self.data)

    def preprocess_data(self, stream):

        def _convert_line(line):
            feat_ids = []
            feat_vals = []
            for x in line:
                feat_id, feat_val = x.split(':')
                feat_ids.append(int(feat_id))
                feat_vals.append(float(feat_val))
            return (torch.as_tensor(feat_ids, dtype=torch.int64),
                    torch.as_tensor(feat_vals, dtype=torch.float32))

        df = pd.read_table(stream, header=None)
        df = df.loc[:, 0].str.split(n=1, expand=True)
        label = df.loc[:, 0].apply(int)
        data = df.loc[:, 1].str.split().apply(_convert_line)
        return data, label

    def collate_fn(self, batch):
        data, label = tuple(zip(*batch))
        _, feat_val = tuple(zip(*data))
        return (torch.stack(feat_val, dim=0).type(torch.long),
                torch.as_tensor(label, dtype=torch.float32).unsqueeze(dim=-1))


def libsvm_input_fn(filepath, batch_size=256, num_threads=1, **kwargs):

    def _input_fn():
        dataset = LIBSVMDataset(filepath)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=num_threads,
                          collate_fn=dataset.collate_fn)

    return _input_fn
