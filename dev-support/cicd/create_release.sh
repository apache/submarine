#!/bin/bash

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

# The script helps making a release.
# You need to specify release version and branch|tag name.
#
# Here are some helpful documents for the release.
# http://www.apache.org/dev/release.html
# http://www.apache.org/dev/release-publishing
# http://www.apache.org/dev/release-signing.html

set -e

BASEDIR="$(dirname "$0")"
. "${BASEDIR}/common_release.sh"
echo "${BASEDIR}/common_release.sh"

if [[ $# -ne 2 ]]; then
  usage
fi

for var in GPG_PASSPHRASE; do
  if [[ -z "${!var}" ]]; then
    echo "You need ${var} variable set"
    exit 1
  fi
done

RELEASE_VERSION="$1"
GIT_TAG="$2"

function compile_src_and_bin() {
  cd ${WORKING_DIR}/submarine
  echo "mvn versions:set -DnewVersion=${RELEASE_VERSION}"
  mvn versions:set -DnewVersion="${RELEASE_VERSION}"
  echo "mvn clean install package -DskipTests -Psrc"
  mvn clean install package -DskipTests -Psrc
  if [[ $? -ne 0 ]]; then
    echo "Build failed. ${BUILD_FLAGS}"
    exit 1
  fi

}

function make_source_package() {
  # create source package
  cd ${WORKING_DIR}/submarine/submarine-dist/target
  cd submarine-dist-*-src
  # remove unneeded dir .github
  rm -rf submarine-dist-${RELEASE_VERSION}-src/.github
  ${TAR} cfz "submarine-dist-${RELEASE_VERSION}-src.tar.gz" "submarine-dist-${RELEASE_VERSION}-src"
  mv "submarine-dist-${RELEASE_VERSION}-src.tar.gz" ${WORKING_DIR}
  echo "Signing the source package"
  cd "${WORKING_DIR}"
  echo "${GPG_PASSPHRASE}" | gpg --passphrase-fd 0 --armor \
    --output "submarine-dist-${RELEASE_VERSION}-src.tar.gz.asc" \
    --detach-sig "${WORKING_DIR}/submarine-dist-${RELEASE_VERSION}-src.tar.gz"
  ${SHASUM} -a 512 "submarine-dist-${RELEASE_VERSION}-src.tar.gz" > \
    "${WORKING_DIR}/submarine-dist-${RELEASE_VERSION}-src.tar.gz.sha512"
}

function make_binary_release() {
  R_DIR_NAME=submarine-dist-${RELEASE_VERSION}-hadoop-2.9
  cd ${WORKING_DIR}/submarine/submarine-dist/target
  mv "${R_DIR_NAME}.tar.gz" ${WORKING_DIR}
  # sign bin package
  cd ${WORKING_DIR}
  echo "${GPG_PASSPHRASE}" | gpg --passphrase-fd 0 --armor \
    --output "${R_DIR_NAME}.tar.gz.asc" \
    --detach-sig "${R_DIR_NAME}.tar.gz"
  ${SHASUM} -a 512 "${R_DIR_NAME}.tar.gz" > \
    "${R_DIR_NAME}.tar.gz.sha512"
}

if [ -d "${WORKING_DIR}/submarine/submarine-dist/target" ]; then
  if $DEBUG_SUBMARINE_SCRIPT; then
    echo "DEBUGGING, skip re-building submarine"
  fi
else
  git_clone
  compile_src_and_bin
fi

make_source_package
make_binary_release

# remove non release files and dirs
echo "Deleting ${WORKING_DIR}/submarine"
rm -rf "${WORKING_DIR}/submarine"
echo "Release files are created under ${WORKING_DIR}"
