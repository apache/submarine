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
import org.junit.Ignore;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert;
import java.util.List;

@Ignore("SUBMARINE-628")
public class homeIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(homeIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: homeIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void homePagination() throws Exception {
    LOG.info("homeIT is commented, because the page is not included in release 0.5.");
    // // Login
    // LOG.info("Login");
    // pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    // pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    // clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    // pollingWait(By.cssSelector("a[routerlink='/workbench/dashboard']"), MAX_BROWSER_TIMEOUT_SEC);

    // LOG.info("Pagination");
    // List<WebElement> changePageIndexButtons = driver.findElements(By.cssSelector("a[class='ant-pagination-item-link ng-star-inserted']"));
    // // 0: open recent: previous page
    // // 1: open recent: next page
    // // 2: news: previous page
    // // 3: news: next page
    // changePageIndexButtons.get(1).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Project2\")]")).size(), 6);
    // changePageIndexButtons.get(1).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Project3\")]")).size(), 6);
    // changePageIndexButtons.get(0).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Project2\")]")).size(), 6);
    // changePageIndexButtons.get(0).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Project1\")]")).size(), 6);
    // changePageIndexButtons.get(3).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Submarine supports yarn 2.7.x 123\")]")).size(), 5);
    // changePageIndexButtons.get(3).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Submarine supports yarn 2.7.x 456\")]")).size(), 5);
    // changePageIndexButtons.get(2).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Submarine supports yarn 2.7.x 123\")]")).size(), 5);
    // changePageIndexButtons.get(2).click();
    // Assert.assertEquals( driver.findElements(By.xpath("//div[contains(text(), \"Submarine supports yarn 2.7.x\")]")).size(), 5);
  }
}
