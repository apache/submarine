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
import org.apache.submarine.integration.pages.ExperimentPage;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;
import sun.rmi.runtime.Log;
import org.apache.submarine.CommandExecutor;
import org.apache.submarine.ProcessData;
import java.io.File;

public class experimentIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(experimentIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: experimentNew");
    driver = WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void experimentNavigation() throws Exception {
    LOG.info("[Testacse: experimentNavigation]");
    // Init the page object
    ExperimentPage experimentPage = new ExperimentPage(driver);
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/dashboard']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to workspace
    LOG.info("url");
    pollingWait(By.xpath("//span[contains(text(), \"Experiment\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(driver.getCurrentUrl(), "http://localhost:8080/workbench/experiment");

    // Test create new experiment
    LOG.info("new experiment");
    experimentPage.newExperimentButtonClick();
    experimentPage.fillMeta("good-e2e-test", "e2e des", "default", "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150", "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0");
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();

    LOG.info("In env");
    experimentPage.envBtnClick();
    experimentPage.fillEnv("ENV_1", "ENV1");
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();

    // Fail due to incorrect spec name
    LOG.info("In spec fail");
    experimentPage.fillTfSpec(1, new String[]{"wrong name"}, new int[]{1}, new int[]{1}, new String[]{"512M"});
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();
    Assert.assertTrue(experimentPage.getErrorNotification().isDisplayed());
    // Successful request
    LOG.info("In spec success");
    experimentPage.deleteSpec();
    Assert.assertEquals(experimentPage.getSpecs(), 0);
    experimentPage.fillTfSpec(2, new String[]{"Ps", "Worker"}, new int[]{1, 1}, new int[]{1, 1}, new String[]{"1024M", "1024M"});
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();
  }
}
