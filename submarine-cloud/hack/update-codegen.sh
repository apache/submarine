#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
CODEGEN_PKG=${CODEGEN_PKG:-$(cd "${SCRIPT_ROOT}"; ls -d -1 ./vendor/k8s.io/code-generator 2>/dev/null || echo $GOPATH/src/k8s.io/code-generator/)}

#cp -R "$(dirname "${BASH_SOURCE[0]}")/../github.com/apache/submarine/submarine-cloud/pkg" "$(dirname "${BASH_SOURCE[0]}")/.."

#exit;

# generate the code with:
# --output-base    because this script should also be able to run inside the vendor dir of
#                  k8s.io/kubernetes. The output-base is needed for the generators to output into the vendor dir
#                  instead of the $GOPATH directly. For normal projects this can be dropped.
bash "${CODEGEN_PKG}"/generate-groups.sh "deepcopy,client,informer,lister" \
  "github.com/apache/submarine/submarine-cloud/pkg/client" \
  "github.com/apache/submarine/submarine-cloud/pkg/apis" \
  "submarine:v1alpha1" \
  --output-base "$(dirname "${BASH_SOURCE[0]}")/.." \
  --go-header-file "${SCRIPT_ROOT}"/hack/boilerplate.go.txt

# To use your own boilerplate text append:
#   --go-header-file "${SCRIPT_ROOT}"/hack/custom-boilerplate.go.txt

cp -R "$(dirname "${BASH_SOURCE[0]}")/../github.com/apache/submarine/submarine-cloud/pkg" "$(dirname "${BASH_SOURCE[0]}")/.."
rm -R "$(dirname "${BASH_SOURCE[0]}")/../github.com/"
