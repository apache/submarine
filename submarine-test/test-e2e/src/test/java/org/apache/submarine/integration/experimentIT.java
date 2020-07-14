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
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.interactions.Actions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;
import sun.rmi.runtime.Log;

import java.io.File;

public class experimentIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(experimentIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: experimentIT");
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
    pollingWait(By.xpath("//span[contains(text(), \"Experiment\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(driver.getCurrentUrl(), "http://localhost:8080/workbench/experiment");

    // Test create new experiment
    LOG.info("new experiment");
    pollingWait(By.xpath("//button[@id='openExperiment']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertTrue(pollingWait(By.xpath("//form"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed());
    WebDriverWait wait = new WebDriverWait( driver, 15);
    // Basic information section
    pollingWait(By.name("experimentName"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Experiment");
    pollingWait(By.name("description"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Project description");
    pollingWait(By.name("namespace"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e namespace");
    pollingWait(By.name("cmd"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("python3 -m e2e cmd");
    pollingWait(By.name("image"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e custom image");
    pollingWait(By.xpath("//button[@id='go']"), MAX_BROWSER_TIMEOUT_SEC).click();
    // env variables section
    LOG.info("in env");
    Assert.assertTrue(pollingWait(By.xpath("//button[@id='env-btn']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed());
    WebElement envBtn = buttonCheck(By.id("env-btn"), MAX_BROWSER_TIMEOUT_SEC);
    envBtn.click();
    wait.until(ExpectedConditions.visibilityOfAllElementsLocatedBy(By.xpath("//input[@name='key0' or name='value0']")));
    pollingWait(By.name("key0"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e key");
    pollingWait(By.name("value0"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e value");

    pollingWait(By.xpath("//button[@id='go']"), MAX_BROWSER_TIMEOUT_SEC).click();
    // Spec section
    LOG.info("in spec");
    WebElement specBtn = wait.until(ExpectedConditions.elementToBeClickable(By.id("spec-btn")));
    specBtn.click();
    pollingWait(By.name("spec0"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e spec");
    pollingWait(By.name("replica0"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("1");
    pollingWait(By.name("cpu0"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("1");
    pollingWait(By.name("memory0"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("512M");
    Assert.assertTrue(pollingWait(By.xpath("//button[@id='go']"), MAX_BROWSER_TIMEOUT_SEC).isEnabled());
//    pollingWait(By.xpath("//button[@id='go']"), MAX_BROWSER_TIMEOUT_SEC).click();
  }
}
