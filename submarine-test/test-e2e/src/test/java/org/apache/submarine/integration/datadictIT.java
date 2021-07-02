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
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.Keys;


public class datadictIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(datadictIT.class);


  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: datadictIT");
    driver =  WebDriverManager.getWebDriver();
    action = new Actions(driver);
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void dataDictTest() throws Exception {
    DataDictPage dataDictPage = new DataDictPage();
    String URL = getURL("http://127.0.0.1", 8080);
    Sidebars sidebars = new Sidebars(URL);
    LoginPage loginPage = new LoginPage();
    String dictCode, newItemCode, newItemName, newPostfix;

    // Login
    LOG.info("Login");
    loginPage.Login();

    // Start Routing & Navigation in data-dict
    LOG.info("Start Routing & Navigation in data-dict");
    sidebars.gotoDataDict();

    // Add button
    LOG.info("[TEST] Add button");
    // // 1. Add --> Ok --> required feedback --> close
    // Add
    Click(dataDictPage.addBtn, MAX_BROWSER_TIMEOUT_SEC);
    // Ok
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);
    // check if the modal doesn't disappear
    Assert.assertEquals(driver.findElements(By.xpath("//div[contains(text(), \"Add\")]")).size(), 1);
    // Close
    Click(dataDictPage.closeBtn, MAX_BROWSER_TIMEOUT_SEC);
    // check if the model disappear
    Assert.assertEquals(driver.findElements(By.xpath("//div[contains(text(), \"Add\")]")).size(), 0);

    // 2. Add --> set input --> close
    // Add
    Click(dataDictPage.addBtn, MAX_BROWSER_TIMEOUT_SEC);
    // Set input
    SendKeys(dataDictPage.dictCodeInput, MAX_BROWSER_TIMEOUT_SEC, "test new dict code");
    SendKeys(dataDictPage.dictNameInput, MAX_BROWSER_TIMEOUT_SEC, "test new dict name");
    SendKeys(dataDictPage.dictDescription, MAX_BROWSER_TIMEOUT_SEC, "test new dict description");
    // close
    Click(dataDictPage.closeBtn, MAX_BROWSER_TIMEOUT_SEC);
    // check there is no new dict
    Assert.assertEquals(driver.findElements(By.xpath("//td[@id='dataDictCodetest new dict code']")).size(), 0);

    // 3. Add --> set input --> ok --> new dict
    // dataDictPage.addNewDict("test new dict code", "test new dict name", "test new dict description", false);
    // Add
    Click(dataDictPage.addBtn, MAX_BROWSER_TIMEOUT_SEC);
    // Set input
    SendKeys(dataDictPage.dictCodeInput, MAX_BROWSER_TIMEOUT_SEC, "test new dict code");
    SendKeys(dataDictPage.dictNameInput, MAX_BROWSER_TIMEOUT_SEC, "test new dict name");
    SendKeys(dataDictPage.dictDescription, MAX_BROWSER_TIMEOUT_SEC, "test new dict description");
    // Ok
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);
    // check there is new dict
    Assert.assertEquals(driver.findElements(By.xpath("//td[@id='dataDictCodetest new dict code']")).size(), 1);

    // Configuration button
    LOG.info("[TEST] PROJECT_TYPE Configuration button");

    // 3. old dict --> More --> Configuration --> Add --> set input --> OK
    dictCode = "PROJECT_TYPE";
    newItemCode = "qqq";
    newItemName = "www";
    // More
    action.moveToElement(driver.findElement(dataDictPage.moreBtn(dictCode))).build().perform();
    // Configuration
    waitToPresent(dataDictPage.configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.configBtn(dictCode))).click().build().perform();
    // Add Item
    waitToPresent(dataDictPage.itemAddBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    Click(dataDictPage.itemAddBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    // Set Input
    Click(dataDictPage.statusDropDown, MAX_BROWSER_TIMEOUT_SEC);
    Click(dataDictPage.statusUnavailable, MAX_BROWSER_TIMEOUT_SEC);
    SendKeys(dataDictPage.itemCodeInput(dictCode), MAX_BROWSER_TIMEOUT_SEC, newItemCode);
    SendKeys(dataDictPage.itemNameInput(dictCode), MAX_BROWSER_TIMEOUT_SEC, newItemName);
    Click(dataDictPage.addActionBtn, MAX_BROWSER_TIMEOUT_SEC);
    // Ok
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);

    // check if add successful
    action.moveToElement(driver.findElement(dataDictPage.moreBtn(dictCode))).build().perform();
    waitToPresent(dataDictPage.configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.configBtn(dictCode))).click().build().perform();
    Assert.assertEquals(driver.findElements(By.xpath(String.format("//td[contains(text(), \"%s\")]", newItemCode))).size(), 1);
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);

    // Edit button
    LOG.info("[TEST] Edit button");

    // 4. Edit dict --> Update --> OK --> More --> Configuration --> OK
    dictCode = "PROJECT_VISIBILITY";
    newPostfix = "123";
    // Edit
    waitToPresent(dataDictPage.editBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    Click(dataDictPage.editBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    // Update
    SendKeys(dataDictPage.dictCodeInput, MAX_BROWSER_TIMEOUT_SEC, newPostfix);
    // Ok
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);
    // More
    action.moveToElement(driver.findElement(dataDictPage.moreBtn(dictCode.concat(newPostfix)))).build().perform();
    // Configuration
    waitToPresent(dataDictPage.configBtn(dictCode.concat(newPostfix)), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.configBtn(dictCode.concat(newPostfix)))).click().build().perform();
    // Ok
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);

    LOG.info("[TEST] test new dict code Configuration button");

    // 5. new dict --> More --> Configuration --> Add --> set input --> OK
    dictCode = "test new dict code";
    newItemCode = "aaa";
    newItemName = "bbb";

    // More
    ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", driver.findElement(dataDictPage.moreBtn(dictCode)));
    waitToPresent(dataDictPage.moreBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.moreBtn(dictCode))).build().perform();
    // Configuration
    waitToPresent(dataDictPage.configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.configBtn(dictCode))).click().build().perform();
    // Add Item
    waitToPresent(dataDictPage.itemAddBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    Click(dataDictPage.itemAddBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    // Set Input
    Click(dataDictPage.statusDropDown, MAX_BROWSER_TIMEOUT_SEC);
    Click(dataDictPage.statusAvailable, MAX_BROWSER_TIMEOUT_SEC);
    SendKeys(dataDictPage.itemCodeInput(dictCode), MAX_BROWSER_TIMEOUT_SEC, newItemCode);
    SendKeys(dataDictPage.itemNameInput(dictCode), MAX_BROWSER_TIMEOUT_SEC, newItemName);
    Click(dataDictPage.addActionBtn, MAX_BROWSER_TIMEOUT_SEC);
    // Ok
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);

    // check if add successful
    action.moveToElement(driver.findElement(dataDictPage.moreBtn(dictCode))).perform();
    waitToPresent(dataDictPage.configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.configBtn(dictCode))).click().build().perform();
    Assert.assertEquals(driver.findElements(By.xpath(String.format("//td[contains(text(), \"%s\")]", newItemCode))).size(), 1);
    Click(dataDictPage.okBtn, MAX_BROWSER_TIMEOUT_SEC);

    // Delete button
    LOG.info("[TEST] Delete button");

    // 6. More --> Delete
    dictCode = "SYS_USER_SEX";
    // More
    action.moveToElement(driver.findElement(dataDictPage.moreBtn(dictCode))).perform();
    // Delete
    waitToPresent(dataDictPage.deleteBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
    action.moveToElement(driver.findElement(dataDictPage.deleteBtn(dictCode))).click().build().perform();
    waitToPresent(dataDictPage.okDeleteBtn, MAX_BROWSER_TIMEOUT_SEC);
    Click(dataDictPage.okDeleteBtn, MAX_BROWSER_TIMEOUT_SEC);
    // check if delete successfully
    Assert.assertEquals(driver.findElements(By.xpath(String.format("//td[@id='dataDictCode%s']", dictCode))).size(), 0);
  }
}
