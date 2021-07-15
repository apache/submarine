---
title: How to Run Frontend Integration Test
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
* The test cases under the directory [test-e2e](https://github.com/apache/submarine/tree/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration) are integration tests to ensure the correctness of the Submarine Workbench.

* These test cases can be run either locally or on GitHub Actions.

## Run E2E test locally

1. Ensure you have setup the submarine locally. If not, you can refer to `Submarine Local Deployment`.

2. Forward port

    ```bash
    kubectl port-forward --address 0.0.0.0 service/submarine-traefik 32080:80
    ```

3. Modify run_frontend_e2e.sh

    You need to modify the port and the URL in this script to where you run the workbench on.

   > Example:
   > If your Submarine workbench is running on 127.0.0.1:4200, you should modify the **WORKBENCH_PORT** to 4200.

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

4. Run run_frontend_e2e.sh (Run a specific test case)

   This script will check whether the port can be accessed or not, and run the test case.
   ```bash
   # at submarine-test/test_e2e
   ./run_fronted_e2e.sh ${TESTCASE}
   # TESTCASE is the IT you want to run, ex: loginIT, experimentIT...
   ```

5. Run all test cases
* Following commands will compile all files and run all files ending with "IT" in the [directory](https://github.com/apache/submarine/tree/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration).
    ```bash
    # Make sure the Submarine workbench is running on 127.0.0.1:8080
    cd submarine/submarine-test/test-e2e
    # Method 1: 
    mvn verify

    # Method 2:
    mvn clean install -U
    ```

## Run E2E test in GitHub Actions

* Each time a commit is pushed, GitHub Actions will be triggered automatically.

## Add a new frontend E2E test case
* **WARNING**
  * You **MUST** read the [document](https://www.selenium.dev/documentation/en/webdriver/waits/) carefully, and understand the difference between **explicit wait**, **implicit wait**, and **fluent wait**.
  * **Do not mix implicit and explicit waits.** Doing so can cause unpredictable wait times.
* We define many useful functions in [AbstractSubmarineIT.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/AbstractSubmarineIT.java).