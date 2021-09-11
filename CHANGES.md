# SUBMARINE Changelog

## Release Notes - Apache Submarine - Version 0.5.0 (2020-12-17)

## Sub-task

- \[[SUBMARINE-502](https://issues.apache.org/jira/browse/SUBMARINE-502)\] - \[WEB\] Implement outline experiment information page.
- \[[SUBMARINE-554](https://issues.apache.org/jira/browse/SUBMARINE-554)\] - \[WEB\] UI for Submarine experiment creation
- \[[SUBMARINE-555](https://issues.apache.org/jira/browse/SUBMARINE-555)\] - \[WEB\] Connect workbench with database for param
- \[[SUBMARINE-556](https://issues.apache.org/jira/browse/SUBMARINE-556)\] - \[WEB\] Connect workbench with database for metric
- \[[SUBMARINE-557](https://issues.apache.org/jira/browse/SUBMARINE-557)\] - Install notebook controller with helm chart
- \[[SUBMARINE-558](https://issues.apache.org/jira/browse/SUBMARINE-558)\] - \[API\] Define Swagger API for pre-defined template submission
- \[[SUBMARINE-559](https://issues.apache.org/jira/browse/SUBMARINE-559)\] - \[API\] Define Swagger API for pre-defined template registration/delete, etc.
- \[[SUBMARINE-560](https://issues.apache.org/jira/browse/SUBMARINE-560)\] - Build an API to manage notebook instances with Swagger
- \[[SUBMARINE-561](https://issues.apache.org/jira/browse/SUBMARINE-561)\] - \[SDK\] Add Pytorch implementation of Attentional Factorization Machine
- \[[SUBMARINE-565](https://issues.apache.org/jira/browse/SUBMARINE-565)\] - Documentation for Environment Management
- \[[SUBMARINE-566](https://issues.apache.org/jira/browse/SUBMARINE-566)\] - \[WEB\] Create a new experiment through UI
- \[[SUBMARINE-568](https://issues.apache.org/jira/browse/SUBMARINE-568)\] - \[WEB\] Fix a display bug in experiment info page
- \[[SUBMARINE-573](https://issues.apache.org/jira/browse/SUBMARINE-573)\] - \[WEB\] Add checkbox in experiment page
- \[[SUBMARINE-575](https://issues.apache.org/jira/browse/SUBMARINE-575)\] - \[SDK\] Set job_name to id in database
- \[[SUBMARINE-579](https://issues.apache.org/jira/browse/SUBMARINE-579)\] - \[WEB\] Add duration information to experiment
- \[[SUBMARINE-582](https://issues.apache.org/jira/browse/SUBMARINE-582)\] - \[WEB\] Implement frontend of environment page in workbench
- \[[SUBMARINE-585](https://issues.apache.org/jira/browse/SUBMARINE-585)\] - \[WEB\] Edit existing experiments through UI
- \[[SUBMARINE-586](https://issues.apache.org/jira/browse/SUBMARINE-586)\] - Delete/Get a notebook instance with REST API
- \[[SUBMARINE-587](https://issues.apache.org/jira/browse/SUBMARINE-587)\] - \[WEB\]The list of notebook instances
- \[[SUBMARINE-590](https://issues.apache.org/jira/browse/SUBMARINE-590)\] - \[WEB\] Update chart/param/metric UI
- \[[SUBMARINE-591](https://issues.apache.org/jira/browse/SUBMARINE-591)\] - \[WEB\] Update notebook component in workbench
- \[[SUBMARINE-596](https://issues.apache.org/jira/browse/SUBMARINE-596)\] - \[WEB\] Implement environment list through UI
- \[[SUBMARINE-597](https://issues.apache.org/jira/browse/SUBMARINE-597)\] - Support for "ssh" based git sync mode
- \[[SUBMARINE-598](https://issues.apache.org/jira/browse/SUBMARINE-598)\] - Support get environment list from database
- \[[SUBMARINE-601](https://issues.apache.org/jira/browse/SUBMARINE-601)\] - Notebook API docs
- \[[SUBMARINE-607](https://issues.apache.org/jira/browse/SUBMARINE-607)\] - \[WEB\] Create and Delete of notebook instances
- \[[SUBMARINE-608](https://issues.apache.org/jira/browse/SUBMARINE-608)\] - \[WEB\] Create environment through UI
- \[[SUBMARINE-609](https://issues.apache.org/jira/browse/SUBMARINE-609)\] - \[WEB\] Delete environment through UI
- \[[SUBMARINE-618](https://issues.apache.org/jira/browse/SUBMARINE-618)\] - Create/Delete IngressRoute with Notebook CR
- \[[SUBMARINE-625](https://issues.apache.org/jira/browse/SUBMARINE-625)\] - \[WEB\] Support running standalone script
- \[[SUBMARINE-629](https://issues.apache.org/jira/browse/SUBMARINE-629)\] - \[E2E_Test\] Improve environmentIT.java
- \[[SUBMARINE-630](https://issues.apache.org/jira/browse/SUBMARINE-630)\] - \[E2E_Test\] Improve notebookIT.java
- \[[SUBMARINE-631](https://issues.apache.org/jira/browse/SUBMARINE-631)\] - \[WEB\] Fix workbench notebook page
- \[[SUBMARINE-632](https://issues.apache.org/jira/browse/SUBMARINE-632)\] - Users only get their own notebook instances
- \[[SUBMARINE-635](https://issues.apache.org/jira/browse/SUBMARINE-635)\] - \[WEB\] Add sync code to experiment UI
- \[[SUBMARINE-636](https://issues.apache.org/jira/browse/SUBMARINE-636)\] - Update user documents for notebook
- \[[SUBMARINE-650](https://issues.apache.org/jira/browse/SUBMARINE-650)\] - \[WEB\] Update the workbench with new notebookSpec
- \[[SUBMARINE-676](https://issues.apache.org/jira/browse/SUBMARINE-676)\] - Add NG-ZORRO UI component library to notebook page

## Bug

- \[[SUBMARINE-538](https://issues.apache.org/jira/browse/SUBMARINE-538)\] - Update the release doc
- \[[SUBMARINE-541](https://issues.apache.org/jira/browse/SUBMARINE-541)\] - Running time is null sometimes
- \[[SUBMARINE-553](https://issues.apache.org/jira/browse/SUBMARINE-553)\] - Modify "set -e" command in some shell scripts
- \[[SUBMARINE-563](https://issues.apache.org/jira/browse/SUBMARINE-563)\] - Liveness probe failed in notebook-controller pod
- \[[SUBMARINE-567](https://issues.apache.org/jira/browse/SUBMARINE-567)\] - Fix travis test failure
- \[[SUBMARINE-576](https://issues.apache.org/jira/browse/SUBMARINE-576)\] - Always return empty list when listing environments
- \[[SUBMARINE-581](https://issues.apache.org/jira/browse/SUBMARINE-581)\] - Change submarine-cloud/bin permission
- \[[SUBMARINE-588](https://issues.apache.org/jira/browse/SUBMARINE-588)\] - \[SDK\] Fix checkstyle error in core.py
- \[[SUBMARINE-589](https://issues.apache.org/jira/browse/SUBMARINE-589)\] - Update the version of virtualenv in CICD docker file
- \[[SUBMARINE-599](https://issues.apache.org/jira/browse/SUBMARINE-599)\] - \[WEB\] Clone experiment through UI
- \[[SUBMARINE-602](https://issues.apache.org/jira/browse/SUBMARINE-602)\] - \[WEB\] Fix some bugs in workspace page
- \[[SUBMARINE-604](https://issues.apache.org/jira/browse/SUBMARINE-604)\] - Fix Double-checked locking in SubmarineConfiguration.java
- \[[SUBMARINE-605](https://issues.apache.org/jira/browse/SUBMARINE-605)\] - Failed to exec method getUserByName in SysUserService class
- \[[SUBMARINE-612](https://issues.apache.org/jira/browse/SUBMARINE-612)\] - Environment variable not found in jupyter pod
- \[[SUBMARINE-615](https://issues.apache.org/jira/browse/SUBMARINE-615)\] - Fix Submarine interpreter test failure
- \[[SUBMARINE-620](https://issues.apache.org/jira/browse/SUBMARINE-620)\] - Fix hard coding of shebang in some scripts
- \[[SUBMARINE-627](https://issues.apache.org/jira/browse/SUBMARINE-627)\] - Can't find the new environment when it named "my-submarine-env"
- \[[SUBMARINE-637](https://issues.apache.org/jira/browse/SUBMARINE-637)\] - Failed to build from sourcec code in centos7.8
- \[[SUBMARINE-656](https://issues.apache.org/jira/browse/SUBMARINE-656)\] - Fix the wrong message when creating k8s object fail
- \[[SUBMARINE-660](https://issues.apache.org/jira/browse/SUBMARINE-660)\] - Remove submarine-workbench/workbench-web/node/node
- \[[SUBMARINE-663](https://issues.apache.org/jira/browse/SUBMARINE-663)\] - Use md5sum command in publish_release.sh

## New Feature

- \[[SUBMARINE-25](https://issues.apache.org/jira/browse/SUBMARINE-25)\] - Manage submarine's docker image with docker hub
- \[[SUBMARINE-230](https://issues.apache.org/jira/browse/SUBMARINE-230)\] - \[Umbrella\] Submarine Interpreter Module
- \[[SUBMARINE-507](https://issues.apache.org/jira/browse/SUBMARINE-507)\] - \[Umbrella\] Submarine Environment Management
- \[[SUBMARINE-570](https://issues.apache.org/jira/browse/SUBMARINE-570)\] - Support run experiment/notebook with synced code
- \[[SUBMARINE-616](https://issues.apache.org/jira/browse/SUBMARINE-616)\] - Deploying Traefik as Ingress controller in Submarine on k8s
- \[[SUBMARINE-633](https://issues.apache.org/jira/browse/SUBMARINE-633)\] - \[SDK\] Support run experiment with synced code
- \[[SUBMARINE-679](https://issues.apache.org/jira/browse/SUBMARINE-679)\] - submarine spark-security plugin needs to support spark3.0

## Improvement

- \[[SUBMARINE-533](https://issues.apache.org/jira/browse/SUBMARINE-533)\] - Add pysubmarine ci dockerfile
- \[[SUBMARINE-550](https://issues.apache.org/jira/browse/SUBMARINE-550)\] - \[doc\] Upload pysubmarine to pypi
- \[[SUBMARINE-574](https://issues.apache.org/jira/browse/SUBMARINE-574)\] - Add a script to initialize database
- \[[SUBMARINE-577](https://issues.apache.org/jira/browse/SUBMARINE-577)\] - Add pypi badge in README
- \[[SUBMARINE-578](https://issues.apache.org/jira/browse/SUBMARINE-578)\] - Update /docs/userdocs/k8s/api/experiment.md
- \[[SUBMARINE-614](https://issues.apache.org/jira/browse/SUBMARINE-614)\] - Environment Id should be string in JSON response
- \[[SUBMARINE-617](https://issues.apache.org/jira/browse/SUBMARINE-617)\] - Update the apache/notebook:jupyter-notebook docker image
- \[[SUBMARINE-619](https://issues.apache.org/jira/browse/SUBMARINE-619)\] - Refactor Readme
- \[[SUBMARINE-621](https://issues.apache.org/jira/browse/SUBMARINE-621)\] - Fix the typo in run_deepfm.py
- \[[SUBMARINE-622](https://issues.apache.org/jira/browse/SUBMARINE-622)\] - Experiment Template API docs
- \[[SUBMARINE-624](https://issues.apache.org/jira/browse/SUBMARINE-624)\] - Replace ingress-nginx with Traefik
- \[[SUBMARINE-628](https://issues.apache.org/jira/browse/SUBMARINE-628)\] - \[WEB\] Disable WIP page link
- \[[SUBMARINE-640](https://issues.apache.org/jira/browse/SUBMARINE-640)\] - Remove workbench (vue)
- \[[SUBMARINE-641](https://issues.apache.org/jira/browse/SUBMARINE-641)\] - Auto convert experiment name to lowercase
- \[[SUBMARINE-642](https://issues.apache.org/jira/browse/SUBMARINE-642)\] - Install latest python sdk in notebook
- \[[SUBMARINE-643](https://issues.apache.org/jira/browse/SUBMARINE-643)\] - Use git-sync image from Docker hub
- \[[SUBMARINE-645](https://issues.apache.org/jira/browse/SUBMARINE-645)\] - Push all image from grc.io to apache docker hub
- \[[SUBMARINE-647](https://issues.apache.org/jira/browse/SUBMARINE-647)\] - Change Workbench user name
- \[[SUBMARINE-648](https://issues.apache.org/jira/browse/SUBMARINE-648)\] - Add a submarine-sdk example in Jupyter notebook
- \[[SUBMARINE-652](https://issues.apache.org/jira/browse/SUBMARINE-652)\] - Autoformat staged files before commit
- \[[SUBMARINE-654](https://issues.apache.org/jira/browse/SUBMARINE-654)\] - \[WEB\] Set default parameter when creating experiment
- \[[SUBMARINE-655](https://issues.apache.org/jira/browse/SUBMARINE-655)\] - Add a deepfm example in Jupyter notebook
- \[[SUBMARINE-658](https://issues.apache.org/jira/browse/SUBMARINE-658)\] - Rename publick-api.ts to public-api.ts
- \[[SUBMARINE-662](https://issues.apache.org/jira/browse/SUBMARINE-662)\] - Clarify duplicate experiment error
- \[[SUBMARINE-674](https://issues.apache.org/jira/browse/SUBMARINE-674)\] - Delete the unused environment "my-submarine-env" in DB
- \[[SUBMARINE-692](https://issues.apache.org/jira/browse/SUBMARINE-692)\] - \[web\] Replace hand-crafted authguard with Angular builtin function

## Test

- \[[SUBMARINE-530](https://issues.apache.org/jira/browse/SUBMARINE-530)\] - \[SDK\] Submarine client e2e test
- \[[SUBMARINE-551](https://issues.apache.org/jira/browse/SUBMARINE-551)\] - Add a new test of LDAP user authentication.
- \[[SUBMARINE-572](https://issues.apache.org/jira/browse/SUBMARINE-572)\] - Create token for login users
- \[[SUBMARINE-583](https://issues.apache.org/jira/browse/SUBMARINE-583)\] - Add unit test for ClusterRestAPI.java
- \[[SUBMARINE-584](https://issues.apache.org/jira/browse/SUBMARINE-584)\] - Add unit test for ExperimentRestApi.java

## Task

- \[[SUBMARINE-545](https://issues.apache.org/jira/browse/SUBMARINE-545)\] - Additional Changes to User Doc (K8s)
- \[[SUBMARINE-592](https://issues.apache.org/jira/browse/SUBMARINE-592)\] - ExperimentRestApiIT#testTensorFlowUsingEnvWithJsonSpec doesn't do expected asserts
- \[[SUBMARINE-670](https://issues.apache.org/jira/browse/SUBMARINE-670)\] - Improvements of Submarine user docs
