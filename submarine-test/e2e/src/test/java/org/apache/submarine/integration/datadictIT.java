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
import org.apache.submarine.SubmarineITUtils;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.interactions.Actions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;
import java.util.*;

public class datadictIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(datadictIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: datadictIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void dataDictAdd() throws Exception {
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/dashboard']"), MAX_BROWSER_TIMEOUT_SEC);

    // Start Routing & Navigation in data-dict 
    LOG.info("Start Routing & Navigation in data-dict");
    pollingWait(By.xpath("//span[contains(text(), \"Manager\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    WebDriverWait wait = new WebDriverWait( driver, 15, 5000);
    pollingWait(By.xpath("//a[@href='/workbench/manager/dataDict']"), MAX_BROWSER_TIMEOUT_SEC).click();
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//span[@class='ant-breadcrumb-link ng-star-inserted']")));
    Assert.assertEquals(driver.getCurrentUrl(), "http://localhost:4200/workbench/manager/dataDict");

    // Add button
    pollingWait(By.cssSelector("form > nz-form-item:nth-child(3) > nz-form-control > div > span > button.ant-btn.ant-btn-default"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Add\")]")).size(), 1);
    pollingWait(By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-default']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Add\")]")).size(), 0);

    // Edit button
    pollingWait(By.cssSelector("submarine-data-dict > nz-card > div > nz-table > nz-spin > div > div > div > div > div > table > tbody > tr:nth-child(1) > td.td-action.ant-table-td-right-sticky > a:nth-child(1)"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Edit\")]")).size(), 1);
    pollingWait(By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-default']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Edit\")]")).size(), 0);
  }
}
