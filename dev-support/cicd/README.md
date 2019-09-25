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
To use them more easily, we provide a Docker image to help committer to handle tasks like committing code and release build.

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
