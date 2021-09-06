#!/usr/bin/env python
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys


def combineJsons(jsonFile1, jsonFile2, outputFile):
    dict1 = json.load(open(jsonFile1))
    dict2 = json.load(open(jsonFile2))
    dict3 = dict(dict1.items() + dict2.items())

    with open(outputFile, "w") as output:
        json.dump(dict3, output, indent=2, sort_keys=True)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise Exception(u"3 arguments needed")

    print(combineJsons(sys.argv[1], sys.argv[2], sys.argv[3]))
