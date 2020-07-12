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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from submarine.utils.fileio import open_buffered_file_reader, file_info

import os
import itertools
import functools
import multiprocessing as mp
from typing import List, Tuple


class LIBSVMDataset(Dataset):

    def __init__(self, data_uri: str, sample_offset: np.ndarray):
        self.data_uri = data_uri
        self.sample_offset = sample_offset

    def __len__(self) -> int:
        return len(self.sample_offset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        with open_buffered_file_reader(self.data_uri) as infile:
            infile.seek(self.sample_offset[idx], os.SEEK_SET)
            sample = infile.readline()
        return LIBSVMDataset.parse_sample(sample)

    @classmethod
    def parse_sample(cls,
                     sample: bytes) -> Tuple[torch.Tensor, torch.Tensor, int]:
        label, *entries = sample.rstrip(b'\n').split(b' ')
        feature_idx = torch.zeros(len(entries), dtype=torch.long)
        feature_value = torch.zeros(len(entries), dtype=torch.float)
        for i, entry in enumerate(entries):
            fidx, fvalue = entry.split(b':')
            feature_idx[i], feature_value[i] = int(fidx), float(fvalue)
        return feature_idx, feature_value, int(label)

    @classmethod
    def prepare_dataset(cls, data_uri: str, n_jobs: int = os.cpu_count()):
        sample_offset = LIBSVMDataset._locate_sample_offsets(data_uri=data_uri,
                                                             n_jobs=n_jobs)
        return LIBSVMDataset(data_uri=data_uri, sample_offset=sample_offset)

    @classmethod
    def _locate_sample_offsets(cls, data_uri: str, n_jobs: int) -> np.ndarray:
        finfo = file_info(data_uri)
        chunk_size, _ = divmod(finfo.size, n_jobs)

        chunk_starts = [0]
        with open_buffered_file_reader(data_uri) as infile:
            while infile.tell() < finfo.size:
                infile.seek(chunk_size, os.SEEK_CUR)
                infile.readline()
                chunk_starts.append(min(infile.tell(), finfo.size))

        with mp.Pool(processes=n_jobs,
                     maxtasksperchild=1) as pool:
            return np.asarray(
                list(
                    itertools.chain.from_iterable(
                        pool.imap(functools.partial(
                            LIBSVMDataset._locate_sample_offsets_job, data_uri),
                                  iterable=enumerate(
                                      zip(chunk_starts[:-1],
                                          chunk_starts[1:]))))))

    @classmethod
    def _locate_sample_offsets_job(
            cls, data_uri: str, task: Tuple[int, Tuple[int, int]]) -> List[int]:
        _, (start, end) = task
        offsets = [start]
        with open_buffered_file_reader(data_uri) as infile:
            infile.seek(start, os.SEEK_SET)
            while infile.tell() < end:
                infile.readline()
                offsets.append(infile.tell())
            assert offsets.pop() == end
        return offsets


def libsvm_input_fn(filepath, batch_size=256, num_threads=1, **kwargs):

    def _input_fn():
        dataset = LIBSVMDataset.prepare_dataset(data_uri=filepath,
                                                n_jobs=num_threads)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=0)  # should be 0 (pytorch bug)

    return _input_fn
