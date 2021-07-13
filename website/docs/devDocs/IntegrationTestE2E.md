---
title: How to Run integration E2E Test
---

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

## Introduction

* It checks the components in the website works correctly.

* You can run the test-e2e either locally or on GitHub Actions.

## Run E2E test locally

1. Ensure you have setup the submarine locally. If not, you can refer to `Submarine Local Deployment`.

2. Forward port

  ```bash
  kubectl port-forward --address 0.0.0.0 service/submarine-traefik 32080:80
  ```

3. Modify run_frontend_e2e.sh

    You need to modify the port and the URL in this script to where you run the workbench on.

   > Example:
   > If you just finished developing the workbench appearance and the workbench is running on localhost:4200, then you should modify the WORKBENCH_PORT to 4200

   ```bash
   # at submarine-test/test_e2e/run_frontend_e2e.sh
   ...
   # ======= Modifiable Variables ======= #
    # Note: URL must start with "http" 
    # (Ref: https://www.selenium.dev/selenium/docs/api/java/org/openqa/selenium/WebDriver.html#get(java.lang.String))
    WORKBENCH_PORT=8080 #<= modify this
    URL="http://127.0.0.1" #<=modify this
    # ==================================== #
    ...
   ```

4. Run run_frontend_e2e.sh

   This script will check whether the port can be accessed or not, and run the test case.
   ```bash
   # at submarine-test/test_e2e
   ./run_fronted_e2e.sh ${TESTCASE}
   # TESTCASE is the IT you want to run, ex: loginIT, experimentIT...
   ```

## Run E2E test in GitHub Actions

Each time a code is submitted, GitHub Actions is automatically triggered for testing.