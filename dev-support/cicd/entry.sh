#!/bin/bash
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
set -e

function start_menu(){
  printf "Menu:\n"
  printf "\t1. Merge PR\n"
  read -p "Enter Menu ID:" menu_id
  case $menu_id in
    "1")
      merge_pr
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

  if [ -z "$JIRA_USERNAME" ]; then
    read -p "Enter Your Apache JIRA User name: "  jira_name
  else
    jira_name=$JIRA_USERNAME
  fi
  echo "Got JIRA name: ${jira_name}"

  if [ -z "$JIRA_PASSWORD" ]; then
    read -s -p "Enter Your Apache JIRA User passwd: "  jira_pwd
  else
    jira_pwd=$JIRA_PASSWORD
  fi

  if [ -z "$APACHE_ID" ]; then
    printf "\n"
    read -p "Enter Your Apache committer ID: "  apache_id
  else
    apache_id=$APACHE_ID
  fi
  echo "Got Apache ID: ${apache_id}"

  if [ -z "$APACHE_NAME" ]; then
    read -p "Enter Your Apache committer name: "  apache_name
  else
    apache_name=$APACHE_NAME
  fi
  echo "Got Apache name: ${apache_name}"

  cd $SUBMARINE_HOME
  git pull
  git config user.name "${apache_name}"
  git config user.email "${apache_id}@apache.org"
  export JIRA_USERNAME=${jira_name}
  export JIRA_PASSWORD=${jira_pwd}
  python dev-support/cicd/merge_submarine_pr.py
  printf "==== Merge PR END ====\n"
}

start_menu
