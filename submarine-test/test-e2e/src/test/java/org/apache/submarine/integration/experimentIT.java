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
import org.apache.submarine.integration.components.Sidebars;
import org.apache.submarine.integration.pages.ExperimentPage;
import org.apache.submarine.integration.pages.LoginPage;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class experimentIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(experimentIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Test case]: experimentNew");
    driver = WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void experimentNavigation() throws Exception {
    String URL = getURL("http://127.0.0.1", 8080);

    LOG.info("[Test case]: experimentNavigation]");
    // Init the page object
    ExperimentPage experimentPage = new ExperimentPage(driver);
    Sidebars sidebars = new Sidebars(URL);
    LoginPage loginPage = new LoginPage();

    // Login
    LOG.info("Login");
    loginPage.Login();

    // Routing to workspace
    LOG.info("url");
    sidebars.gotoExperiment();

    // Test create new experiment
    LOG.info("First step");
    experimentPage.newExperimentButtonClick();
    experimentPage.customizedBtnClick();
    experimentPage.advancedButtonCLick();
    experimentPage.envBtnClick();
    String experimentName = "experiment-e2e-test";
    experimentPage.fillExperimentMeta(experimentName, "e2e des", "default",
            "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log" +
                    " --learning_rate=0.01 --batch_size=150",
            "apache/submarine:tf-mnist-with-summaries-1.0",
            "ENV_1", "ENV1");
    SendKeys(By.xpath("//input[@id='git-repo']"), MAX_BROWSER_TIMEOUT_SEC, "https://github.com/apache/submarine.git");
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();

    LOG.info("Second step");
    // Fail due to incorrect spec name
    LOG.info("In spec fail");
    experimentPage.fillTfSpec(1, new String[]{"Master"}, new int[]{-1}, new int[]{-1}, new int[]{512});
    Assert.assertFalse(experimentPage.getGoButton().isEnabled());
    // Successful request
    LOG.info("In spec success");
    experimentPage.deleteSpec();
    experimentPage.fillTfSpec(2, new String[]{"Ps", "Worker"}, new int[]{1, 1}, new int[]{1, 1}, new int[]{1024, 1024});
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();

    LOG.info("Preview experiment spec");
    Assert.assertTrue(experimentPage.getGoButton().isEnabled());
    experimentPage.goButtonClick();
  }

  /*
      TODO: Launch submarine server and K8s in e2e-test
      Comment out because of Experiment creation failure on Travis

      @Test
      public void updateExperiment() {
        ....
      }
  */

}
