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
import org.apache.submarine.integration.pages.LoginPage;
import org.apache.submarine.WebDriverManager;
import org.apache.submarine.integration.components.Sidebars;
import org.junit.Ignore;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert;

@Ignore("SUBMARINE-628")
public class interpreterIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(interpreterIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: interpreterIT");
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
<<<<<<< HEAD
      // Login
>>>>>>> login in the current non-used page
=======
    // Login
>>>>>>> fix indent and remove comment
    LOG.info("Login");
    loginPage.Login();

    // Routing to Interpreter
    sidebars.gotoInterpreter();

    // Test create new Interpreter
    pollingWait(By.xpath("//button[@id='interpreterAddButton']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//input[@id='inputNewInterpreterName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("Python Interpreter 2");
    pollingWait(By.xpath("//input[@id='inputNewInterpreterType']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("Python");
    pollingWait(By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-primary']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals( driver.findElements(By.xpath("//td[@id='interpreterNamePython Interpreter 2']")).size(), 1);
  }
}
