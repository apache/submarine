#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
# activate python 2.7.13 environment
. ${PYTHON_VENV_PATH}/venv2.7/bin/activate

function start_menu(){
  printf "Menu:\n"
  printf "\t1. Merge PR\n"
  printf "\t2. Update Submarine Website\n"
  read -p "Enter Menu ID:" menu_id
  case $menu_id in
    "1")
      merge_pr
    ;;
    "2")
      update_submarine_site
    ;;
    "*")
      printf "unknown. Exiting."
    ;;
  esac
}

function merge_pr(){
  printf "==== Merge PR Begin ====\n"
  jira_name="n"
  jira_pwd="p"
  apache_id="id"
  apache_name="name"

  if [ -z "${JIRA_USERNAME:-}" ]; then
    read -p "Enter Your Apache JIRA User name: "  jira_name
  else
    jira_name=$JIRA_USERNAME
  fi
  echo "Got JIRA name: ${jira_name}"

  if [ -z "${JIRA_PASSWORD:-}" ]; then
    read -s -p "Enter Your Apache JIRA User passwd: "  jira_pwd
  else
    jira_pwd=$JIRA_PASSWORD
  fi

  if [ -z "${APACHE_ID:-}" ]; then
    printf "\n"
    read -p "Enter Your Apache committer ID: "  apache_id
  else
    apache_id=$APACHE_ID
  fi
  echo "Got Apache ID: ${apache_id}"

  if [ -z "${APACHE_NAME:-}" ]; then
    read -p "Enter Your Apache committer name: "  apache_name
  else
    apache_name=$APACHE_NAME
  fi
  echo "Got Apache name: ${apache_name}"

  cd $SUBMARINE_HOME
  git checkout master
  git pull
  git config user.name "${apache_name}"
  git config user.email "${apache_id}@apache.org"
  export JIRA_USERNAME=${jira_name}
  export JIRA_PASSWORD=${jira_pwd}
  python dev-support/cicd/merge_submarine_pr.py
  printf "==== Merge PR END ====\n"
}

function update_submarine_site(){
  printf "==== Update Submarine Site Begin ====\n"
  apache_id="id"
  apache_name="name"

  if [ -z "${APACHE_ID:-}" ]; then
    printf "\n"
    read -p "Enter Your Apache committer ID: "  apache_id
  else
    apache_id=$APACHE_ID
  fi
  echo "Got Apache ID: ${apache_id}"

  if [ -z "${APACHE_NAME:-}" ]; then
    read -p "Enter Your Apache committer name: "  apache_name
  else
    apache_name=$APACHE_NAME
  fi
  echo "Got Apache name: ${apache_name}"

  cd $SUBMARINE_SITE
  git checkout master
  git pull
  git config user.name "${apache_name}"
  git config user.email "${apache_id}@apache.org"
  git config credential.helper store
  bundle update
  bundle exec jekyll serve --watch --host=0.0.0.0 > /tmp/jekyll.log 2>&1 &
  echo "==== Please use vim to edit md files and check http://localhost:4000/ for the update ===="
  while true; do
    echo "==== Edit Mode: Type 'exit' when you finish the changes (you don't need to perform git commit/push). ===="
    bash
    read -p "Have you finished updating the MD files? y/n/quit " commit
    case $commit in
      "y")
        echo "Start committing changes.."
        cd $SUBMARINE_SITE
        git add .
        git status
        read -p "Please input the commit message: " message
        git commit -m "${message} (master branch)"
        git push origin master
        cp -r _site /_site
        git checkout asf-site
        cp -r /_site/* ./
        git add .
        git status
        git commit -m "${message} (asf-site branch)"
        git push origin asf-site
        echo "Exiting edit mode.."
        break
      ;;
      "n")
        continue
      ;;
      "quit")
        printf "Exiting edit mode.."
        break
      ;;
      "q")
        printf "Exiting edit mode.."
        break
      ;;
      "*")
        printf "Unknown. Exiting edit mode.."
        break
      ;;
    esac
  done
  printf "\n"
  printf "==== Update Submarine Site END ====\n"
  echo "==== Enter shell again incase any unexpected error happens ===="
  bash
  echo "Exiting CICD.."
}

start_menu
deactivate
