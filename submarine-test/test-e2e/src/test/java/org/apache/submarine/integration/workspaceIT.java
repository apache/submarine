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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;

public class workspaceIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(workspaceIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: workspaceIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void workspaceNavigation() throws Exception {
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/dashboard']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to workspace
    pollingWait(By.xpath("//span[contains(text(), \"Workspace\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(driver.getCurrentUrl(), "http://localhost:8080/workbench/workspace");

    //Test project part
    pollingWait(By.xpath("//li[contains(text(), \"Project\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='addProjectbtn']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//div[@id='addProjectbtn']/button"), MAX_BROWSER_TIMEOUT_SEC).click();
    //step1
    Assert.assertEquals(pollingWait(By.xpath("//form"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//input[@id='username']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Project");
    pollingWait(By.xpath("//textarea[@name='projectDescriptin']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Project description");
    pollingWait(By.xpath("//div[@class='centerDiv']/button"), MAX_BROWSER_TIMEOUT_SEC).click();
    //step2
    Assert.assertEquals(pollingWait(By.xpath("//nz-tabset"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//div[@class='centerDiv']/button[last()]"), MAX_BROWSER_TIMEOUT_SEC).click();
    //step3
    Assert.assertEquals(pollingWait(By.xpath("//thead"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//div[@class='centerDiv']/button[last()-1]"), MAX_BROWSER_TIMEOUT_SEC).click();
    //return to project page
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='addProjectbtn']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);

    WebDriverWait wait = new WebDriverWait( driver, 60);
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//li[contains(text(), \"Release\")]")));

    //Test release part
    pollingWait(By.xpath("//li[contains(text(), \"Release\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//nz-table[@id='releaseTable']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);

    //Test training part
    pollingWait(By.xpath("//li[contains(text(), \"Training\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='trainingDiv']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);

    //Test team part
    pollingWait(By.xpath("//li[contains(text(), \"Team\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='teamDiv']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);


    // shared part
    pollingWait(By.xpath("//li[contains(text(), \"Shared\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//nz-table[@id='sharedTable']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    
  }
}

