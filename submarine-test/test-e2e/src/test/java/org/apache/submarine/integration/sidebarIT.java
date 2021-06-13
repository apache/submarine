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

public class sidebarIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(sidebarIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: sidebarIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void sidebarNavigation() throws Exception {
    String URL = getURL("http://127.0.0.1", 8080);
    // Login
    Login();

    // Start Routing & Navigation in sidebar
    LOG.info("Start Routing & Navigation in sidebar");
    ClickAndNavigate(By.xpath("//span[contains(text(), \"Experiment\")]"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/workbench/experiment"));
    Click(By.xpath("//span[contains(text(), \"Manager\")]"), MAX_BROWSER_TIMEOUT_SEC);
    Click(By.xpath("//a[@href='/workbench/manager/user']"), MAX_BROWSER_TIMEOUT_SEC);

    // Lazy-loading
    ClickAndNavigate(By.xpath("//a[@href='/workbench/manager/user']"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/workbench/manager/user"));
    ClickAndNavigate(By.xpath("//a[@href='/workbench/manager/dataDict']"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/workbench/manager/dataDict"));
  }
}
