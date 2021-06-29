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
import org.apache.submarine.integration.pages.LoginPage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;

public class loginIT extends AbstractSubmarineIT {
  public final static Logger LOG = LoggerFactory.getLogger(loginIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: loginIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void loginUser() throws Exception {
    // Testcase1
    LOG.info("[Sub-Testcase-1] Invalid User");
    LOG.info("Enter blank username and password");

    // Init the page object
    LoginPage loginPage = new LoginPage(driver);

    // Click sign in without input
    loginPage.clickSignInBtn();

    // Assert the warning texts
    Assert.assertEquals(driver.findElements(loginPage.getUserNameSignInWarning()).size(), 1);
    Assert.assertEquals(driver.findElements(loginPage.getPasswordSignInWarning()).size(), 1);

    LOG.info("Enter invalid username and password");

    loginPage.fillLoginForm("123", "123");

    loginPage.clickSignInBtn();

    loginPage.waitForWarningPresent();

    loginPage.fillLoginForm("\b\b\b", "\b\b\b");

    // Testcase2
    LOG.info("[Sub-Testcase-2] Valid User");
    loginPage.Login();
  }
}
