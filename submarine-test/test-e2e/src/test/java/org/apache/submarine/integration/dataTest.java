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
import org.apache.submarine.integration.components.Sidebars;
import org.apache.submarine.WebDriverManager;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.Assert;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@Ignore("SUBMARINE-628")
public class dataTest extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(dataTest.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: dataIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }


  @Test
  public void dataNavigation() throws Exception {
    String URL = getURL("http://127.0.0.1", 8080);

    Sidebars sidebars = new Sidebars(URL);

    // Login
    LoginPage loginPage = new LoginPage();

    LOG.info("Login");
    loginPage.Login();

    // Routing to data page
    sidebars.gotoData();

    // Test create new Table
    pollingWait(By.xpath("//button[@id='createBtn']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertTrue(pollingWait(By.xpath("//form"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed());

    pollingWait(By.xpath("//button[@id='firstNextBtn']"), MAX_BROWSER_TIMEOUT_SEC).click();
    pollingWait(By.xpath("//input[@id='tableName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("e2e test Table");
    pollingWait(By.xpath("//button[@id='secondNextBtn']"), MAX_BROWSER_TIMEOUT_SEC).click();

    pollingWait(By.xpath("//button[@id='submit']"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertTrue(pollingWait(By.xpath("//thead"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed());
  }
}
