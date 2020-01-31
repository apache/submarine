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

if [ -z "$ROOT" ]; then
    echo "error: ROOT should be initialized"
    exit 1
fi

OS=$(go env GOOS)
ARCH=$(go env GOARCH)
OUTPUT=${ROOT}/output
OUTPUT_BIN=${OUTPUT}/bin
KUBECTL_VERSION=1.12.10
KUBECTL_BIN=$OUTPUT_BIN/kubectl
KIND_VERSION=0.6.0
KIND_BIN=$OUTPUT_BIN/kind

test -d "$OUTPUT_BIN" || mkdir -p "$OUTPUT_BIN"

function hack::verify_kubectl() {
    if test -x "$KUBECTL_BIN"; then
        [[ "$($KUBECTL_BIN version --client --short | grep -o -E '[0-9]+\.[0-9]+\.[0-9]+')" == "$KUBECTL_VERSION" ]]
        return
    fi
    return 1
}

function hack::ensure_kubectl() {
    if hack::verify_kubectl; then
        return 0
    fi
    echo "Installing kubectl v$KUBECTL_VERSION..."
    tmpfile=$(mktemp)
    trap "test -f $tmpfile && rm $tmpfile" RETURN
    curl --retry 10 -L -o $tmpfile https://storage.googleapis.com/kubernetes-release/release/v${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl
    mv $tmpfile $KUBECTL_BIN
    chmod +x $KUBECTL_BIN
}

function hack::verify_kind() {
    if test -x "$KIND_BIN"; then
        [[ "$($KIND_BIN --version 2>&1 | cut -d ' ' -f 3)" == "$KIND_VERSION" ]]
        return
    fi
    return 1
}

function hack::ensure_kind() {
    if hack::verify_kind; then
        return 0
    fi
    echo "Installing kind v$KIND_VERSION..."
    tmpfile=$(mktemp)
    trap "test -f $tmpfile && rm $tmpfile" RETURN
    curl --retry 10 -L -o $tmpfile https://github.com/kubernetes-sigs/kind/releases/download/v${KIND_VERSION}/kind-$(uname)-amd64
    mv $tmpfile $KIND_BIN
    chmod +x $KIND_BIN
}

# hack::version_ge "$v1" "$v2" checks whether "v1" is greater or equal to "v2"
function hack::version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

function hack::check_requirements() {
    # Check requirements
    for requirement in kind docker kubectl
    do
        echo "############ check ${requirement} ##############"
        if hash ${requirement} 2>/dev/null;then
            echo "${requirement} have installed"
        else
            echo "this script needs ${requirement}, please install ${requirement} first."
            if test ${requirement} = "kind"; then
                hack::ensure_kind
                echo "Please add $KIND_BIN to PATH variable or copy it to one of the locations $PATH"
            fi
            if test ${requirement} = "kubectl"; then
                hack::ensure_kubectl
                echo "Please add $KUBECTL_BIN to PATH variable or copy it to one of the locations $PATH"
            fi
            exit 1
        fi
    done
}
