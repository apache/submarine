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

Build the image and start the container

```
./build_and_start_cicd_image.sh
```

Or separate image building and container running process

```
./build.sh
docker run -it --rm apache/submarine:cicd-0.7.0
```

Or directly use the image on docker hub

```
docker run -it --rm apache/submarine:cicd-0.7.0
```

After successfully running the container, you will see output like below and then you can decide what to accomplish.

```
Start Submarine CI/CD.
==== Merge PR Begin ====
Enter Your Apache JIRA User name:
```

Jira username, password, apache id and apache username can be provided to the docker container so that you do not need to type them everytime. You can add them to the local environment variable file.

For example, create a file named `~/.secrets/submarine-cicd.env` and write the following content to it.

```
JIRA_USERNAME=YOUR_JIRA_USERNAME
JIRA_PASSWORD=YOUR_JIRA_PASSWORD
APACHE_ID=YOUR_APACHE_ID
APACHE_NAME=YOUR_APACHE_NAME
```

And then you can run the docker image with the following command

```
docker run -it --env-file ~/.secrets/submarine-cicd.env -p 4000:4000 --rm apache/submarine:cicd-0.7.0
```

## Manual mode (Not Recommended)

First, You need to have `python 3` and run `pip insall jira`

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
11. ./dev-support/cicd/merge_submarine_pr.py
