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
<<<<<<< HEAD
import org.apache.submarine.integration.components.Sidebars;
=======
import org.apache.submarine.integration.pages.LoginPage;
>>>>>>> login in the current non-used page
import org.apache.submarine.WebDriverManager;
import org.junit.Ignore;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert;

@Ignore("SUBMARINE-628")
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
    String URL = getURL("http://127.0.0.1", 8080);
<<<<<<< HEAD
    Sidebars sidebars = new Sidebars(URL);

    // Login
=======
    LoginPage loginPage = new LoginPage(driver);
      // Login
>>>>>>> login in the current non-used page
    LOG.info("Login");
    loginPage.Login();
//    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
//    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
//    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
//    pollingWait(By.cssSelector("a[routerlink='/workbench/experiment']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to workspace
    sidebars.gotoWorkSpace();

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

    //Test project part
    pollingWait(By.xpath("//li[contains(text(), \"Project\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//div[@id='addProjectbtn']")));
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='addProjectbtn']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//div[@id='addProjectbtn']/button"), MAX_BROWSER_TIMEOUT_SEC).click();
    //step1
    By nextStepButton = By.xpath("//div[@class='centerDiv']/button");
    Assert.assertEquals(pollingWait(By.xpath("//form"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//input[@id='username']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Project");
    pollingWait(By.xpath("//textarea[@name='projectDescription']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Project description");
    Assert.assertEquals(pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).getAttribute("disabled"), "true"); // nextStepButton disabled (no set visibility)

    pollingWait(By.xpath("//nz-radio-group[@name='visibility']/label[1]/span/input"), MAX_BROWSER_TIMEOUT_SEC).click(); // select Private
    Assert.assertEquals(pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).getAttribute("disabled"), null); // nextStepButton enabled

    pollingWait(By.xpath("//nz-radio-group[@name='visibility']/label[last()]/span/input"), MAX_BROWSER_TIMEOUT_SEC).click(); // select Public
    Assert.assertEquals(pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).getAttribute("disabled"), "true"); // nextStepButton disabled (no set permission)

    pollingWait(By.xpath("//nz-radio-group[@name='permission']/label[last()]/span/input"), MAX_BROWSER_TIMEOUT_SEC).click(); // select Can View
    Assert.assertEquals(pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).getAttribute("disabled"), null); // nextStepButton enabled

    pollingWait(By.xpath("//nz-radio-group[@name='visibility']/label[2]/span/input"), MAX_BROWSER_TIMEOUT_SEC).click(); // select Team
    Assert.assertEquals(pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).getAttribute("disabled"), "true"); // nextStepButton disabled (no set Team)

    pollingWait(By.xpath("//nz-select[@name='team']"), MAX_BROWSER_TIMEOUT_SEC).click(); // expand team options
    pollingWait(By.xpath("//li[@nz-option-li][last()]"), MAX_BROWSER_TIMEOUT_SEC).click(); // select a team
    Assert.assertEquals(pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).getAttribute("disabled"), null); // nextStepButton enabled

    pollingWait(nextStepButton, MAX_BROWSER_TIMEOUT_SEC).click();

    //step2
    Assert.assertEquals(pollingWait(By.xpath("//nz-tabset"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//div[@class='centerDiv']/button[last()]"), MAX_BROWSER_TIMEOUT_SEC).click();

    //step3
    Assert.assertEquals(pollingWait(By.xpath("//thead"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    pollingWait(By.xpath("//div[@class='centerDiv']/button[last()-1]"), MAX_BROWSER_TIMEOUT_SEC).click();
    //return to project page
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='addProjectbtn']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);



  }
}

