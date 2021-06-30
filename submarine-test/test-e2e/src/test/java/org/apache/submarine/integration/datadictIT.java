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
import org.apache.submarine.integration.pages.DataDictPage;
import org.apache.submarine.WebDriverManager;
import org.openqa.selenium.By;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

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

  @Test //TODO(kevin85421): Due to the undeterministic behavior of travis, I decide to comment it.
  public void dataDictTest() throws Exception {
    DataDictPage dataDictPage = new DataDictPage();
    String URL = getURL("http://127.0.0.1", 8080);

    Sidebars sidebars = new Sidebars(URL);
    LoginPage loginPage = new LoginPage();

    // Login
    LOG.info("Login");
    loginPage.Login();

    // Start Routing & Navigation in data-dict
    LOG.info("Start Routing & Navigation in data-dict");
    sidebars.gotoDataDict();
    WebDriverWait wait = new WebDriverWait( driver, 60);
    wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//span[@class='ant-breadcrumb-link ng-star-inserted']")));

    // Add button
    LOG.info("[TEST] Add button");
    // Add --> Ok --> required feedback --> close
    dataDictPage.addNothing();

    // Add --> set input --> close
    dataDictPage.addNewDict("test new dict code", "test new dict name", "test new dict description", true);

    // Add --> set input --> ok --> new dict
    dataDictPage.addNewDict("test new dict code", "test new dict name", "test new dict description", false);


    // Configuration button
    LOG.info("[TEST] PROJECT_TYPE Configuration button");

    // old dict --> More --> Configuration --> Add --> set input --> OK
    dataDictPage.addItem("PROJECT_TYPE","qqq", "www", false);


    // Edit button
    LOG.info("[TEST] Edit button");

    // Edit dict --> Update --> OK --> More --> Configuration
    dataDictPage.editDict("PROJECT_VISIBILITY", "123");

    LOG.info("[TEST] test new dict code Configuration button");

    // new dict --> More --> Configuration --> Add --> set input --> OK
    dataDictPage.addItem("test new dict code", "aaa", "bbb", true);

    // Delete button
    LOG.info("[TEST] Delete button");

    // More --> Delete
    dataDictPage.deleteDict("SYS_USER_SEX");
  }
}
