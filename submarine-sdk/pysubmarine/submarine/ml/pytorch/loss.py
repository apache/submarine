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

from torch.nn.modules import loss


def get_loss_fn(key):
    def _condition_fn(x):
        k, v = x
        return isinstance(v, type) and issubclass(v, loss._Loss) and (k.lower() == key.lower())
    _, loss_fn = next(iter(filter(_condition_fn, vars(loss).items())))
    return loss_fn
