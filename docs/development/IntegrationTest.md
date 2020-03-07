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

# IntegrationTest

Submarine now supports two kinds of integration tests.

They are in the project's `submarine/submarine-test` directory, There are two modules, `e2e` and `test-k8s`.

There are currently some differences between `test-e2e` and `test-k8s` in operation mode.

Among them, `test-e2e` needs to deploy submarine locally, while `test-k8s` uses k8s to deploy submarine.

These two different test methods can be applied to different test scenarios. (In the future, these two test methods may be combined or adjusted)

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

## e2e test

e2e tests can be ran both locally and in Travis

Local testing: When developers perform e2e testing locally, they need to manually start the submarine server by executing bin / submarine-daemon.sh.

Then you can manually runs test cases in the `test-e2e/test` directory in IDEA.

### Run the existing tests.
##### Move to the working directroy.
```
cd submarine/submarine-test/test-e2e
```
##### Compile & Run.

> Following command will compile all files and run all files ending with "IT".

**If your workbench server is not working on port 32777 ([mini-submarine](https://github.com/apache/submarine/tree/master/dev-support/mini-submarine) maps the workbench port 8000 to 32777), please first modify the port in WebDriverManager.java line 61  to the port where your workbench run.**

*   Execute the following command in your host machine to get the port
```
docker inspect --format='{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' mini-submarine
```

*   For linux
 ```
 mvn verify
 ```

*   For MacOS
```
mvn clean install -U
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
1. Create new file ending with "IT" under "submarine/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration/".
2. Your public class is recommended to extend AbstractSubmarineIT. The class AbstractSubmarineIT contains some commonly used functions.
```java
  WebElement pollingWait(final By locator, final long timeWait); // Find element on the website.
  void clickAndWait(final By locator); // Click element and wait for 1 second.
  void sleep(long millis, boolean logOutput); // Let system sleep a period of time.
```
3. There are also some commonly used functions except in AbstractSybmarineIT.java.
```java
  // In WebDriverManager.java:
  public static WebDriver getWebDriver(); // This return a firefox webdriver which has been set to your workbench website.
```
4. Add [JUnit](https://junit.org/junit5/docs/current/user-guide/) annotation before your testing function, e.g., @Beforeclass, @Test, and @AfterClass. You can refer to [loginIT.java](https://github.com/apache/submarine/blob/master/submarine-test/test-e2e/src/test/java/org/apache/submarine/integration/loginIT.java).
5. Use command mentioned above to compile and run to test whether it works as your anticipation.


