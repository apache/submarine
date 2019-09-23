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
# Introduction
This cicd directory contains several scripts to ease code and release management.

## Merge PRs From Github
The "merge_submarine_pr.py" script is usd for merge PRs without access the github web portal. It can also automatically merge activities from github to apache Jira.

As a committer, you should create a dedicated directory to do below steps instead of using existing development repo:

1. git clone https://gitbox.apache.org/repos/asf/hadoop-submarine.git
2. cd hadoop-submarine
3. git remote rename origin apache
4. git remote add apache-github https://github.com/apache/hadoop-submarine.git
5. git config --local --add user.name {name}
6. git config --local --add user.email {username}@apache.org
7. echo -e "JIRA_USERNAME={jira_username}\nJIRA_PASSWORD={jira_password}" >> ~/.bashrc
8. source ~/.bashrc
9. dev-support/cicd/merge_submarine_pr.py
