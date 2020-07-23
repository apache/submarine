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

package org.apache.submarine.integration.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.PageFactory;
import org.openqa.selenium.support.pagefactory.AjaxElementLocatorFactory;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.util.List;

public class ExperimentPage {

  @FindBy(id = "go")
  private WebElement goButton;

  @FindBy(id = "openExperiment")
  private WebElement newExperimentButton;

  /*
  * For svg/path/g element tag, we must use //*[name = 'svg'] to select
  * //svg will fail
  */
  @FindBy(xpath = "//*[name() = 'svg' and @data-icon = 'close-circle']")
  private List<WebElement> deleteBtns;

  // Meta form
  @FindBy(name = "experimentName")
  private WebElement experimentName;

  @FindBy(name = "description")
  private WebElement description;

  @FindBy(name = "namespace")
  private WebElement namespace;

  @FindBy(name = "cmd")
  private WebElement cmd;

  @FindBy(name = "image")
  private WebElement image;

  // Env form
  @FindBy(id = "env-btn")
  private WebElement envBtn;

  @FindBy(xpath = "//input[contains(@name, 'key')]")
  private WebElement envKey;

  @FindBy(xpath = "//input[contains(@name, 'value')]")
  private WebElement envValue;

  // Spec
  @FindBy(id = "spec-btn")
  private WebElement specBtn;

  @FindBy(xpath = "//input[contains(@name, 'spec')]")
  private List<WebElement> specNames;

  @FindBy(xpath = "//input[contains(@name, 'replica')]")
  private List<WebElement> replicas;

  @FindBy(xpath = "//input[contains(@name, 'cpu')]")
  private List<WebElement> cpus;

  @FindBy(xpath = "//input[contains(@name, 'memory')]")
  private List<WebElement> memory;

  // Notification
  @FindBy(xpath = "//div[contains(@class, 'ant-message-error')]//span")
  private WebElement errorNotification;

  @FindBy(xpath = "//div[contains(@class, 'ant-message-success')]//span")
  private WebElement successNotification;

  private WebDriverWait wait;

  public ExperimentPage(WebDriver driver) {
    // NoSuchElementException will be thrown if the elements are not found
    PageFactory.initElements(new AjaxElementLocatorFactory(driver, 5), this);
    wait = new WebDriverWait(driver, 15);
  }

  // Getter
  public WebElement getGoButton() {
    return goButton;
  }

  public WebElement getErrorNotification() {
    return errorNotification;
  }


  public int getSpecs() {
    return specNames.size();
  }

  // button click actions
  public void goButtonClick() {
    wait.until(ExpectedConditions.elementToBeClickable(goButton)).click();
  }

  public void newExperimentButtonClick() {
    wait.until(ExpectedConditions.elementToBeClickable(newExperimentButton)).click();
  }

  public void envBtnClick() {
    wait.until(ExpectedConditions.elementToBeClickable(envBtn)).click();
  }

  public void specBtnClick() {
    wait.until(ExpectedConditions.elementToBeClickable(specBtn)).click();
  }


  // Real actions
  public void fillMeta(String name, String des, String namespaceStr, String cmdStr, String imageStr) {
    experimentName.clear();
    experimentName.sendKeys(name);
    description.clear();
    description.sendKeys(des);
    namespace.clear();
    namespace.sendKeys(namespaceStr);
    cmd.clear();
    cmd.sendKeys(cmdStr);
    image.clear();
    image.sendKeys(imageStr);
  }

  public void fillEnv(String key, String value) {
    envKey.sendKeys(key);
    envValue.sendKeys(value);
  }

  public void deleteSpec() {
    for (WebElement d : deleteBtns) {
      d.click();
    }
  }

  public void fillTfSpec(int specCount, String[] inputNames, int[] replicaCount, int[] cpuCount, String[] inputMemory) {
    for (int i = 0; i < specCount; i++) {
      specBtnClick();
    }

    for (int i = 0; i < specCount; i++) {
      specNames.get(i).sendKeys(inputNames[i]);
      replicas.get(i).sendKeys(Integer.toString(replicaCount[i]));
      cpus.get(i).sendKeys(Integer.toString(cpuCount[i]));
      memory.get(i).sendKeys(inputMemory[i]);
    }

  }
}
