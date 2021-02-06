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

import org.apache.submarine.AbstractSubmarineIT;
import org.apache.submarine.WebDriverManager;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;


public class environmentIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(experimentIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: environmentIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void experimentNavigation() throws Exception {
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/dashboard']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to workspace
    LOG.info("url");
    pollingWait(By.xpath("//span[contains(text(), \"Environment\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(driver.getCurrentUrl(), "http://localhost:8080/workbench/environment");

    // Test create new environment
    LOG.info("Create new environment");
    pollingWait(By.xpath("//button[@id='btn-newEnvironment']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//button[@id='btn-cancel']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//button[@id='btn-newEnvironment']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.cssSelector("input[ng-reflect-name='environmentName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testEnvName");
    pollingWait(By.cssSelector("input[ng-reflect-name='dockerImage']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testDockerImage");
    pollingWait(By.cssSelector("input[ng-reflect-name='name']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testName");
    pollingWait(By.xpath("//button[@id='addChannel-btn']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//input[@id='channel0']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testChannel");
    pollingWait(By.xpath("//button[@id='addDep-btn']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//input[@id='dependencies0']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("testDep");
    pollingWait(By.xpath("//button[@id='btn-submit']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//td[contains(., 'testEnvName')]"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    Thread.sleep(2000);
  }
}
