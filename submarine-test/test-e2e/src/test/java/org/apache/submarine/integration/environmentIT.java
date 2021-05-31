/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.integration;

import org.apache.commons.io.FileUtils;
import org.apache.submarine.AbstractSubmarineIT;
import org.apache.submarine.WebDriverManager;
import org.openqa.selenium.By;
import org.openqa.selenium.support.ui.FluentWait;
import org.openqa.selenium.support.ui.Wait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

import static java.util.concurrent.TimeUnit.SECONDS;


public class environmentIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(experimentIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: environmentIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown() throws IOException {
    File dir = new File(WebDriverManager.getDownloadPath());
    FileUtils.cleanDirectory(dir);
    driver.quit();
  }

  @Test
  public void environmentNavigation() throws Exception {
    String URL = getURL("http://localhost", 8080);
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/experiment']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to workspace
    LOG.info("url");
    pollingWait(By.xpath("//span[contains(text(), \"Environment\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(driver.getCurrentUrl(), URL.concat("/workbench/environment"));

    // Test create new environment
    LOG.info("Create new environment");
    Assert.assertEquals(pollingWait(By.xpath("//button[@id='btn-newEnvironment']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//button[@id='btn-newEnvironment']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.cssSelector("input[ng-reflect-name='environmentName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testEnvName");
    pollingWait(By.cssSelector("input[ng-reflect-name='dockerImage']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testDockerImage");
    pollingWait(By.xpath("//nz-upload[@id='upload-config']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.cssSelector("input[type=file]"), MAX_BROWSER_TIMEOUT_SEC).sendKeys(System.getProperty("user.dir") + "/src/test/resources/test_config_1.yml");
    Assert.assertEquals(pollingWait(By.xpath("//button[@id='btn-submit']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//button[@id='btn-submit']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//button[@id='btn-newEnvironment']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);

    // Test download environment spec
    pollingWait(By.xpath("//a[@id='btn-downloadEnvironmentSpec0']"), MAX_BROWSER_TIMEOUT_SEC).click();
    File fileToCheck = Paths.get(WebDriverManager.getDownloadPath()).resolve("environmentSpec.json").toFile();
    Wait wait = new FluentWait(driver).withTimeout(MAX_BROWSER_TIMEOUT_SEC, SECONDS);
    wait.until(WebDriver -> fileToCheck.exists());
    LOG.info("Test done.");
  }
}
