#!/usr/bin/env bash                                                        

#
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
#

pip install yamllint

helm template helm-charts/submarine --output-dir _yamllint_helm_charts
YAMLLINT_OUTPUT=$(yamllint . -c .yamllint)
echo "$YAMLLINT_OUTPUT"
YAMLLINT_RETURN=$?
rm -r _yamllint_helm_charts

if test $YAMLLINT_RETURN -ne 0; then
    echo -e "yamllint checks failed with 1 or more errors.\n"
    exit 1
else
    echo -e "Checkstyle checks passed."
fi
