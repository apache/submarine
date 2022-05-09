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
import org.apache.submarine.integration.components.Sidebars;
import org.apache.submarine.WebDriverManager;
import org.apache.submarine.integration.pages.LoginPage;
import org.apache.submarine.SubmarineITUtils;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

public class notebookTest extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(notebookTest.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: notebookIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void notebookNavigation() throws Exception {
    String URL = getURL("http://127.0.0.1", 8080);
    Sidebars sidebars = new Sidebars(URL);
    // Login
    LoginPage loginPage = new LoginPage();
    loginPage.Login();

    // Routing to Notebook
    sidebars.gotoNoteBook();

    // Test for creating new notebook
    LOG.info("Create Notebook Test");
    Click(By.xpath("//button[@id='btn-newNotebook']"), MAX_BROWSER_TIMEOUT_SEC);
    SendKeys(By.cssSelector("input[ng-reflect-name='notebookName']"), MAX_BROWSER_TIMEOUT_SEC, "test-nb");
    SendKeys(By.cssSelector("input[ng-reflect-name='cpus']"), MAX_BROWSER_TIMEOUT_SEC,"2");
    SendKeys(By.cssSelector("input[ng-reflect-name='gpus']"), MAX_BROWSER_TIMEOUT_SEC, "1");
    SendKeys(By.cssSelector("input[ng-reflect-name='memoryNum']"), MAX_BROWSER_TIMEOUT_SEC, "1024");
    Click(By.xpath("//button[@id='envVar-btn']"), MAX_BROWSER_TIMEOUT_SEC);
    SendKeys(By.xpath("//input[@name='key0']"), MAX_BROWSER_TIMEOUT_SEC, "testKey0");
    SendKeys(By.xpath("//input[@name='value0']"), MAX_BROWSER_TIMEOUT_SEC, "testValue0");
    Click(By.xpath("//button[@id='nb-form-btn-create']"), MAX_BROWSER_TIMEOUT_SEC);
    /*
    Future add k8s test.
    Assert.assertEquals(pollingWait(By.xpath("//td[contains(., 'test-nb')]"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);
    */
    waitToPresent(By.xpath("//button[@id='btn-newNotebook']"), MAX_BROWSER_TIMEOUT_SEC);
    LOG.info("Test Success!");

  }
}
