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
import org.openqa.selenium.By;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;

public class registerIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(registerIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: registerIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void registerFrontEndInvalidTest() throws Exception {
    String URL = getURL("http://127.0.0.1", 8080);
    // Navigate from Login page to Registration page
    LOG.info("Navigate from Login page to Registration page");
    ClickAndNavigate(By.xpath("//a[contains(text(), \"Create an account!\")]"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/user/register"));

    // Username test
    //   Case1: empty username
    SendKeys(By.cssSelector("input[formcontrolname='username']"), MAX_BROWSER_TIMEOUT_SEC, " \b");
    waitToPresent(By.xpath("//div[contains(text(), \"Enter your username!\")]"), MAX_BROWSER_TIMEOUT_SEC);
    //   Case2: existed username
    SendKeys(By.cssSelector("input[formcontrolname='username']"), MAX_BROWSER_TIMEOUT_SEC, "test");
    waitToPresent(By.xpath("//div[contains(text(), \"The username already exists!\")]"), MAX_BROWSER_TIMEOUT_SEC);

    // Email test
    //   Case1: empty email
    SendKeys(By.cssSelector("input[formcontrolname='email']"), MAX_BROWSER_TIMEOUT_SEC, " \b");
    waitToPresent(By.xpath("//div[contains(text(), \"Type your email!\")]"), MAX_BROWSER_TIMEOUT_SEC);
    //   Case2: existed email
    String existedEmailTestCase = "test@gmail.com";
    SendKeys(By.cssSelector("input[formcontrolname='email']"), MAX_BROWSER_TIMEOUT_SEC, existedEmailTestCase);
    waitToPresent(By.xpath("//div[contains(text(), \"The email is already used!\")]"), MAX_BROWSER_TIMEOUT_SEC); 
    //   Case3: invalid email
    String backspaceKeys = "";
    for ( int i=0; i < (existedEmailTestCase.length() - existedEmailTestCase.indexOf("@")); i++) {
        backspaceKeys += "\b";
    };
    SendKeys(By.cssSelector("input[formcontrolname='email']"), MAX_BROWSER_TIMEOUT_SEC, backspaceKeys);
    waitToPresent(By.xpath("//div[contains(text(), \"The email is invalid!\")]"), MAX_BROWSER_TIMEOUT_SEC); 
    
    // Password test
    //   Case1: empty password
    SendKeys(By.cssSelector("input[formcontrolname='password']"), MAX_BROWSER_TIMEOUT_SEC, " \b");
    waitToPresent(By.xpath("//div[contains(text(), \"Type your password!\")]"), MAX_BROWSER_TIMEOUT_SEC);
    //   Case2: string length must be in 6 ~ 20 characters
    SendKeys(By.cssSelector("input[formcontrolname='password']"), MAX_BROWSER_TIMEOUT_SEC, "testtesttesttesttesttest"); // length = 24
    waitToPresent(By.xpath("//div[contains(text(), \"Password's length must be in 6 ~ 20 characters.\")]"), MAX_BROWSER_TIMEOUT_SEC);
    SendKeys(By.cssSelector("input[formcontrolname='password']"), MAX_BROWSER_TIMEOUT_SEC, "\b\b\b\b\b\b\b\b\b\b\b\b"); // length = 12
     Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Password's length must be in 6 ~ 20 characters.\")]")).size(), 0);

    // Re-enter password test
    //   Case1: empty re-enter password
    SendKeys(By.cssSelector("input[formcontrolname='checkPassword']"), MAX_BROWSER_TIMEOUT_SEC, " \b");
    waitToPresent(By.xpath("//div[contains(text(), \"Type your password again!\")]"), MAX_BROWSER_TIMEOUT_SEC);
    //   Case2: re-enter password != password    
    SendKeys(By.cssSelector("input[formcontrolname='checkPassword']"), MAX_BROWSER_TIMEOUT_SEC, "1234"); // "1234" != "testtesttest"
    waitToPresent(By.xpath("//div[contains(text(), \"Passwords must match!\")]"), MAX_BROWSER_TIMEOUT_SEC);
    ClickAndNavigate(By.xpath("//a[@href='/user/login']"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/user/login"));
  }

  @Test
  public void registerFrontEndValidTest() throws Exception {
    String URL = getURL("http://127.0.0.1", 8080);
    // Sign-Up successfully
    ClickAndNavigate(By.xpath("//a[contains(text(), \"Create an account!\")]"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/user/register"));
    SendKeys(By.cssSelector("input[formcontrolname='username']"), MAX_BROWSER_TIMEOUT_SEC, "validusername");
    SendKeys(By.cssSelector("input[formcontrolname='email']"), MAX_BROWSER_TIMEOUT_SEC, "validemail@gmail.com");
    SendKeys(By.cssSelector("input[formcontrolname='password']"), MAX_BROWSER_TIMEOUT_SEC, "validpassword");
    SendKeys(By.cssSelector("input[formcontrolname='checkPassword']"), MAX_BROWSER_TIMEOUT_SEC, "validpassword");
    Click(By.cssSelector("label[formcontrolname='agree']"), MAX_BROWSER_TIMEOUT_SEC);
    ClickAndNavigate(By.cssSelector("button[class='ant-btn ant-btn-primary ant-btn-block']"), MAX_BROWSER_TIMEOUT_SEC, URL.concat("/user/login")); 
  }
}
