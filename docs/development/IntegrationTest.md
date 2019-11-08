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

## Run the existing tests.
##### Move to the working directroy.
```
cd submarine/submarine-test/e2e
```
##### Compile & Run.

> Following command will compile all files and run all files ending with "IT". 

**If your workbench server is not working on port 32777 ([mini-submarine](https://github.com/apache/submarine/tree/master/dev-support/mini-submarine) maps the workbench port 8000 to 32777), please first modify the port in WebDriverManager.java line 61  to the port where your workbench run.** 

For linux
```
mvn verify
```
For MacOS
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

## Add your own integration test
1. Create new file ending with "IT" under "submarine/submarine-test/e2e/src/test/java/org/apache/submarine/integration/".
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
4. Add [JUnit](https://junit.org/junit5/docs/current/user-guide/) annotation before your testing function, e.g., @Beforeclass, @Test, and @AfterClass. You can refer to [loginIT.java](https://github.com/apache/submarine/blob/master/submarine-test/e2e/src/test/java/org/apache/submarine/integration/loginIT.java).
5. Use command mentioned above to compile and run to test whether it works as your anticipation.


