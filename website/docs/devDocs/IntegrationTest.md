---
title: How to Run Integration Test
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
1.  Now, Apache Submarine supports two kinds of integration test: `test-e2e` and `test-k8s`. These two modules can be found in the [submarine/submarine-test](https://github.com/apache/submarine/tree/master/submarine-test) directory.

2.  Currently, there are some differences between `test-e2e` and `test-k8s` in operation mode. To elaborate, `test-e2e` needs to deploy Apache Submarine locally, while `test-k8s` deploys Apache Submarine via k8s.

3.  These two test modules can be applied to different test scenarios. (In the future, these two test modules may be combined or adjusted)

## k8s test

k8s test: When the user submits the code to his/her repository or the `apache/submarine` git repository, the travis test task will automatically start.

test-k8s runs test cases in travis. It will first create a k8s cluster by using the kind tool in travis,

and then compile and package the submarine project in `submarine-dist` directory to build a docker image.

Then use this latest code to build a docker image and deploy a submarine system in k8s. Then run test case in the `test-k8s/..` directory.

### Run k8s test in locally

Executing the following command will perform the following actions:

```
mvn -Phadoop-2.9 clean package install -DskipTests verify -DskipRat -am -pl submarine-test/test-k8s
```

1. The submarine project will be compiled and packaged to generate `submarine-dist/target/submarine-<version>.tar.gz`
2. Call the `submarine-cloud/hack/integration-test.sh` script

    + Call the `build.sh` script under `submarine/dev-support/docker-images/` to generate the latest `submarine`, `database` and `operator` docker images.
    + Call `submarine-cloud/hack/kind-cluster-build.sh` to create a k8s cluster
    + Call `submarine-cloud/hack/deploy-submarine.sh` to deploy the submarine system in the k8s cluster using the latest `submarine`, `database` and `operator` docker images.
    + Call the test cases in `submarine-test/test-k8s/` for testing.

### Run k8s test in travis

Each time a code is submitted, travis is automatically triggered for testing.

## E2E test

### E2E tests can be executed both locally and in Travis (For workbench developer)
* Run E2E tests locally:
  * Step1: Follow [HowToRun.md](https://github.com/apache/submarine/blob/master/docs/workbench/HowToRun.md) to launch the submarine-server and database.
  * Step2: Run workbench (Angular version) locally
  ```
  cd submarine/submarine-workbench/workbench-web
  npm start
  // Check 127.0.0.1:4200
  ```
  * Step3: Modify the port from 8080 to 4200
    * [WebDriverManager.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/WebDriverManager.java): `url = "http://localhost:8080";` --> `url = "http://localhost:4200";`
    * [Your Unit test case](https://github.com/apache/submarine/tree/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration): `8080` --> `4200`
  * Step4: Comment the `headless` option
    * [ChromeWebDriverProvider.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/ChromeWebDriverProvider.java): `chromeOptions.addArguments("--headless");` --> `//chromeOptions.addArguments("--headless");`
    * With the `headless` option, the selenium will be executed in background.
  * Step5: Run E2E test cases (Please check the following section **Run the existing tests**)
* Run E2E tests in Travis:
  *  Step1: Make sure that the port must be 8080 rather than in [WebDriverManager.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/WebDriverManager.java) and [all test cases](https://github.com/apache/submarine/tree/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration).
  *  Step2: Make sure that the `headless` option is not commented in [ChromeWebDriverProvider.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/ChromeWebDriverProvider.java).
  *  Step3: If you push the commit to Github, the Travis CI will execute automatically and you can check it in `https://travis-ci.com/${your_github_account}/${your_repo_name}`.
### Run the existing tests.
##### Move to the working directory.
```
cd submarine/submarine-test/test-e2e
```
##### Compile & Run.

> Following command will compile all files and run **all** files ending with "IT" in the [directory](https://github.com/apache/submarine/tree/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration).
*   For linux
 ```
 mvn verify
 ```
*   For MacOS
```
mvn clean install -U
```
> Run a specific testcase
```
mvn -Dtest=${your_test_case_file_name} test //ex: mvn -Dtest=loginIT test 
```

##### Result
If all of the function under test are succeeded, it will show.
```
BUILD SUCCESS
```
Otherwise, it will show.
```
BUILD FAILURE
```

### Add your own integration test
1. Create a new file ending with "IT" under "submarine/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration/".
2. Your public class is recommended to extend AbstractSubmarineIT. The class AbstractSubmarineIT contains some commonly used functions.
```java
  WebElement pollingWait(final By locator, final long timeWait); // Find element on the website.
  void clickAndWait(final By locator); // Click element and wait for 1 second.
  void sleep(long millis, boolean logOutput); // Let system sleep a period of time.
```
3. There are also some commonly used functions except in AbstractSubmarineIT.java.
```java
  // In WebDriverManager.java:
  public static WebDriver getWebDriver(); // This return a firefox webdriver which has been set to your workbench website.
```
4. Add [JUnit](https://junit.org/junit5/docs/current/user-guide/) annotation before your testing function, e.g., @Beforeclass, @Test, and @AfterClass. You can refer to [loginIT.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration/loginIT.java).
5. Use command mentioned above to compile and run to test whether it works as your anticipation.


