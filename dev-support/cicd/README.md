<!---
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->
# CICD Introduction

> Please note that cicd is a tool provided to submarine committers for PR merging and release. Only submarine committers have permission to execute.

This cicd directory contains several scripts to ease code and release management. For how-to-release, check [here](./HowToRelease.md)

To use them more easily, we provide a Docker image to help committer to handle tasks like committing code and release build.

## Docker mode

```
./build_and_start_cicd_image.sh
```

Or

```
cd <path-to-submarine-home>/dev-support/cicd
docker build -t submarine-cicd .
docker run -it --rm submarine-cicd
```

Jira username, password, apache id and apache username are required in the docker container. You can
add them to the local environment variable.

```
vi ~/.bash_profile
export JIRA_USERNAME='Your jira username'
export JIRA_PASSWORD='Your jira password'
export APACHE_ID='Your apache id' # Your apache email prefix
export APACHE_NAME='Your apache name'
```

And you'll see output like below and then you can decide what to accomplish.
```
$ docker run -it -e JIRA_USERNAME="${JIRA_USERNAME}" -e JIRA_PASSWORD="${JIRA_PASSWORD}" -e APACHE_ID="${APACHE_ID}" -e APACHE_NAME="${APACHE_NAME}" -p 4000:4000 --rm submarine-cicd
```

The screen outputs the following information: 

```
Menu:
	1. Merge PR
	2. Update Submarine Website
Enter Menu ID:
```

As you can see, the Docker mode support several features like merging PR and updating the web site. Choose the task you need to do and follow the popup tip to go on.

## Manual mode (Not Recommended)

First, You need install `python 2.7.13` and `pip insall jira`

### The Procedure of Merging PR

1. mkdir ${work_dir}
2. cd ${work_dir}
3. git clone https://gitbox.apache.org/repos/asf/submarine.git
4. cd submarine
5. git remote rename origin apache
6. git remote add apache-github https://github.com/apache/submarine.git
7. optional: git config --local --add user.name {name} 
8. optional: git config --local --add user.email {username}@apache.org
9. optional: echo -e "JIRA_USERNAME={jira_username}\nJIRA_PASSWORD={jira_password}" >> ~/.bashrc
10. optional: source ~/.bashrc
11. dev-support/cicd/merge_submarine_pr.py

