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

This cicd directory contains several scripts to ease code and release management.
To use them more easily, we provide a Docker image to help committer to handle tasks like committing code and release build.

## Docker mode

```
cd <path-to-submarine-home>/dev-support/cicd
docker build -t submarine-cicd .
docker run -it --rm submarine-cicd
```

Or

```
./build_and_start_cicd_image.sh
```

And you'll see output like below and then you can decide what to accomplish.
```
$ docker run -it --rm submarine-cicd
Menu:
	1. Merge PR
Enter Menu ID:1
==== Merge PR Begin ====
Enter Your Apache JIRA User name:
```

## Manual mode

### The Procedure of Merging PR

1. mkdir ${work_dir} & cd ${work_dir}
2. git clone https://git-wip-us.apache.org/repos/asf/hadoop-submarine
3. cd hadoop-submarine
4. git remote rename origin apache
5. git remote add apache-github https://github.com/apache/hadoop-submarine.git
6. optional: git config --local --add user.name {name} 
7. optional: git config --local --add user.email {username}@apache.org
8. optional: echo -e "JIRA_USERNAME={jira_username}\nJIRA_PASSWORD={jira_password}" >> ~/.bashrc
9. optional: source ~/.bashrc
10. dev-support/cicd/merge_submarine_pr.py
