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

import org.junit.Assert;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.PageFactory;
import org.openqa.selenium.support.pagefactory.AjaxElementLocatorFactory;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ExperimentPage {

  @FindBy(id = "experimentData")
  private WebElement dataSection;

  @FindBy(id = "go")
  private WebElement goBtn;

  @FindBy(id = "advancedBtn")
  private WebElement advancedBtn;

  @FindBy(id = "openExperiment")
  private WebElement newExperimentBtn;

  @FindBy(id = "customized")
  private WebElement customizedBtn;
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

  @FindBy(id = "namespace")
  private WebElement namespace;

  @FindBy(name = "cmd")
  private WebElement cmd;

  @FindBy(id = "image")
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

  @FindBy(xpath = "//div[contains(@id, 'spec')]//span")
  private List<WebElement> specNamesSelect;

  @FindBy(xpath = "//input[contains(@name, 'replica')]")
  private List<WebElement> replicas;

  @FindBy(xpath = "//input[contains(@name, 'cpu')]")
  private List<WebElement> cpus;

  @FindBy(xpath = "//input[contains(@name, 'memory')]")
  private List<WebElement> memory;

  @FindBy(css = "ul.ant-select-dropdown-menu-root")
  private WebElement dropDownMenu;

  // Notification
  @FindBy(xpath = "//div[contains(@class, 'ant-message-error')]//span")
  private WebElement errorNotification;

  @FindBy(xpath = "//div[contains(@class, 'ant-message-success')]//span")
  private WebElement successNotification;

  public final static Logger LOG = LoggerFactory.getLogger(ExperimentPage.class);
  private WebDriverWait wait;
  private Actions action;

  public ExperimentPage(WebDriver driver) {
    PageFactory.initElements(new AjaxElementLocatorFactory(driver, 10), this);
    wait = new WebDriverWait(driver, 15);
    action = new Actions(driver);
  }

  // Getter
  public WebElement getGoButton() {
    return goBtn;
  }

  // button click actions
  public void goButtonClick() {
    wait.until(ExpectedConditions.elementToBeClickable(goBtn)).click();
  }

  public void advancedButtonCLick() {
    wait.until(ExpectedConditions.elementToBeClickable(advancedBtn)).click();
  }

  public void newExperimentButtonClick() {
    wait.until(ExpectedConditions.elementToBeClickable(newExperimentBtn)).click();
  }

  public void customizedBtnClick() {
    wait.until(ExpectedConditions.elementToBeClickable(customizedBtn)).click();
  }

  public void envBtnClick() {
    wait.until(ExpectedConditions.elementToBeClickable(envBtn)).click();
  }

  public void specBtnClick() {
    wait.until(ExpectedConditions.elementToBeClickable(specBtn)).click();
  }

  // Real actions
  public void fillExperimentMeta(String name, String description, String namespace, String cmd, String image,
              String envKey, String envValue) {
    this.experimentName.clear();
    this.experimentName.sendKeys(name);
    this.description.clear();
    this.description.sendKeys(description);
    Select objSelect = new Select(this.namespace);
    objSelect.selectByVisibleText(namespace);
    this.cmd.clear();
    this.cmd.sendKeys(cmd);
    this.image.clear();
    this.image.sendKeys(image);
    this.envKey.sendKeys(envKey);
    this.envValue.sendKeys(envValue);
  }

  public void deleteSpec() {
    for (WebElement d : deleteBtns) {
      d.click();
    }
  }

  public void fillTfSpec(int specCount, String[] inputNames,
                         int[] replicaCount, int[] cpuCount, int[] inputMemory) {
    for (int i = 0; i < specCount; i++) {
      specBtnClick();
    }


    for (int i = 0; i < specCount; i++) {
      // Click the dropdown menu
      action.moveToElement(specNamesSelect.get(i)).click().perform();
      wait.until(ExpectedConditions.visibilityOf(dropDownMenu));
      String optionStr = ".//li[text()[contains(.,'" + inputNames[i] + "')]]";
      WebElement option = dropDownMenu.findElement(By.xpath(optionStr));
      action.moveToElement(option).click().perform();
      // Rest of the inputs
      replicas.get(i).clear();
      replicas.get(i).sendKeys(Integer.toString(replicaCount[i]));
      cpus.get(i).clear();
      cpus.get(i).sendKeys(Integer.toString(cpuCount[i]));
      memory.get(i).sendKeys(Integer.toString(inputMemory[i]));
    }
  }

  public void editTfSpec(String targetName) {
    String nameFieldStr = "//td[text()[contains(.,'" + targetName + "')]]";
    String editButtonStr = ".//tbody" + nameFieldStr + "//following-sibling::td[@class='td-action']/a[1]";
    LOG.info(editButtonStr);
    WebElement editButton = dataSection.findElement(By.xpath(editButtonStr));
    editButton.click();
    goButtonClick();
    goButtonClick();
    for (WebElement m : memory) {
      m.clear();
      m.sendKeys("512");
    }

    Assert.assertTrue(getGoButton().isEnabled());
    goButtonClick();
    /*
      Must wait until the whole page is loaded, or StaleElementReference will be thrown
      dataSection element is destroyed due to page reloading
     */
    wait.until(ExpectedConditions.visibilityOf(dataSection));
    editButton = dataSection.findElement(By.xpath(editButtonStr));
    wait.until(ExpectedConditions.elementToBeClickable(editButton));
    editButton.click();
    goButtonClick();
    goButtonClick();
    for (WebElement m : memory) {
     String v = m.getAttribute("value");
     Assert.assertEquals(v, "512");
    }
  }
}
