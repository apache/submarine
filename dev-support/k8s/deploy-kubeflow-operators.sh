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
set -euo pipefail

readonly TF_OPERATOR_IMAGE="apache/submarine:tf_operator-v1.1.0-g92389064"
readonly PYTORCH_OPERATOR_IMAGE="apache/submarine:pytorch-operator-v1.1.0-gd596e904"
readonly TF_MNIST_IMAGE="apache/submarine:tf-mnist-with-summaries-1.0"
readonly PT_MNIST_IMAGE="apache/submarine:pytorch-dist-mnist-1.0"

if [ -L "${BASH_SOURCE-$0}" ]; then
  PWD=$(dirname "$(readlink "${BASH_SOURCE-$0}")")
else
  PWD=$(dirname "${BASH_SOURCE-$0}")
fi
CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)
export CURRENT_PATH
export SUBMARINE_HOME=${CURRENT_PATH}/../..
# lib.sh use the ROOT variable
export ROOT="${SUBMARINE_HOME}/submarine-cloud/"
export KUBECONFIG="${HOME}/.kube/kind-config-${clusterName:-kind}"

# shellcheck source=./../../submarine-cloud/hack/lib.sh
source "${SUBMARINE_HOME}/submarine-cloud/hack/lib.sh"

###########################################
# Load local docker image into registry
# Globals:
#   KIND_BIN
# Arguments:
#   image
###########################################
function load_image_to_registry() {
  if [[ ! $(docker inspect "$1" > /dev/null) ]] ; then
    docker pull "$1"
  fi
  ${KIND_BIN} load docker-image "$1"
}

###########################################
# Deploy tf-operator on K8s
# Globals:
#   KUBECTL_BIN
#   CURRENT_PATH
#   TF_OPERATOR_IMAGE
# Arguments:
#   useSample
###########################################
function deploy_tf_operator() {
  load_image_to_registry "${TF_OPERATOR_IMAGE}"

  ${KUBECTL_BIN} apply -f "${CURRENT_PATH}/tfjob/crd.yaml"
  ${KUBECTL_BIN} kustomize "${CURRENT_PATH}/tfjob/operator" \
    | ${KUBECTL_BIN} apply -f -

  if [[ ${1:-} == "true" ]]; then
    load_image_to_registry "${TF_MNIST_IMAGE}"
  fi
}

###########################################
# Deploy tf-operator on K8s
# Globals:
#   KUBECTL_BIN
#   CURRENT_PATH
#   PYTORCH_OPERATOR_IMAGE
# Arguments:
#   useSample
###########################################
function deploy_pytorch_operator() {
  load_image_to_registry "${PYTORCH_OPERATOR_IMAGE}"
  ${KUBECTL_BIN} apply -f "${CURRENT_PATH}/pytorchjob"

  if [[ ${1:-} == "true" ]]; then
    load_image_to_registry "${PT_MNIST_IMAGE}"
  fi
}

###########################################
# Print the usage information
###########################################
function usage() {
  cat <<END

This script aims to deploy the machine learning operator to K8s.

Usage:
    $0 [options]

Options:
  -a,  --all         deploy the TensorFlow and PyTorch operator
  -tf, --tensorflow  deploy the TensorFlow operator
  -pt, --pytorch     deploy the PyTorch operator
  -s,  --sample      pull the sample docker image and load into K8s registry
  -h,  --help        prints this usage message
END
}

function main() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    case $1 in
      -a|--all)
        opt_all="true"
        shift ;;
      -tf|--tensorflow)
        opt_tf="true"
        shift ;;
      -pt|--pytorch)
        opt_pt="true"
        shift ;;
      -s|--sample)
        opt_s="true"
        shift ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown options: $*"
        usage
        exit 2
        ;;
    esac
  done

  opt_all=${opt_all:-}
  opt_tf=${opt_tf:-}
  opt_pt=${opt_pt:-}
  opt_s=${opt_s:-}
  hack::ensure_kubectl

  if [[ "${opt_tf}" == "true" && "${opt_pt}" == "true" ]]; then
    opt_all="true"
  fi
  if [[ "${opt_all}" == "true" ]]; then
    deploy_tf_operator ${opt_s}
    deploy_pytorch_operator ${opt_s}
  elif [[ "${opt_tf}" == "true" ]]; then
    deploy_tf_operator ${opt_s}
  elif [[ "${opt_pt}" == "true" ]]; then
    deploy_pytorch_operator ${opt_s}
  fi
}

main "$@"
