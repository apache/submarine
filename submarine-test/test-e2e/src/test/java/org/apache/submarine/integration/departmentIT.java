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
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;

public class departmentIT extends AbstractSubmarineIT{

  public final static Logger LOG = LoggerFactory.getLogger(departmentIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: departmentIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void dataNavigation() throws Exception {
    String URL = getURL("http://localhost", 8080);
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/experiment']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to department page
    pollingWait(By.xpath("//span[contains(text(), \"Manager\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    WebDriverWait wait = new WebDriverWait( driver, 60);
    pollingWait(By.xpath("//a[@href='/workbench/manager/department']"), MAX_BROWSER_TIMEOUT_SEC).click();
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//span[@class='ant-breadcrumb-link ng-star-inserted']")));
    Assert.assertEquals(driver.getCurrentUrl(), URL.concat("/workbench/manager/department"));

    // Test create new department
    pollingWait(By.xpath("//button[@id='btnAddDepartment']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//input[@id='codeInput'] "), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e Test");
    pollingWait(By.xpath("//input[@id='nameInput']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e Test");
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//label[@id='validCode']")));
    pollingWait(By.xpath("//button[@id='btnSubmit']"), MAX_BROWSER_TIMEOUT_SEC).click();
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//nz-table[@id='departmentTable']")));
    Assert.assertEquals(pollingWait(By.xpath("//td[contains(., 'e2e Test')]"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    
  }
}
