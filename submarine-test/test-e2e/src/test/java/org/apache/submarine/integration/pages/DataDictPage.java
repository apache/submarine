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

import org.apache.submarine.AbstractSubmarineIT;
import org.testng.Assert;
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

public class DataDictPage extends AbstractSubmarineIT {

    private By addBtn = By.cssSelector("form > nz-form-item:nth-child(3) > nz-form-control > div > span > button.ant-btn.ant-btn-default");

    private By okBtn = By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-primary']");

    private By okDeleteBtn = By.xpath("//span[text()='Ok']/ancestor::button[@ng-reflect-nz-type='primary']");

    private By closeBtn = By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-default']");

    private By dictCodeInput = By.xpath("//input[@id='inputNewDictCode']");

    private By dictNameInput = By.xpath("//input[@id='inputNewDictName']");

    private By dictDescription = By.xpath("//input[@id='inputNewDictDescription']");

    // for configuration jump out modal
    private By statusDropDown = By.xpath("//span[@class='ant-cascader-picker-label']");

    private By statusAvailable = By.xpath("//li[@title='available']");

    private By statusUnavailable = By.xpath("//li[@title='unavailable']");

    private By addActionBtn = By.xpath("//button[@class='ant-btn ng-star-inserted ant-btn-default ant-btn-sm']");
    
    public By moreBtn(String dict_code) {
        String xpath = String.format("//a[@id='dataDictMore%s']", dict_code);
        return By.xpath(xpath);
    }

    public By deleteBtn(String dict_code) {
        String xpath = String.format("//li[@id='dataDictDelete%s']", dict_code);
        return By.xpath(xpath);
    }

    public By configBtn(String dict_code) {
        String xpath = String.format("//li[@id='dataDictConfiguration%s']", dict_code);
        return By.xpath(xpath);
    }

    public By editBtn(String dict_code) {
        String xpath = String.format("//a[@id='dataDictEdit%s']", dict_code);
        return By.xpath(xpath);
    }

    public By itemAddBtn(String dict_code) {
        String xpath = String.format("//button[@id='dataDictItemAdd%s']", dict_code);
        return By.xpath(xpath);
    }

    public By itemCodeInput(String dict_code) {
        String xpath = String.format("//input[@id='newItemCode%s']", dict_code);
        return By.xpath(xpath);
    }

    public By itemNameInput(String dict_code) {
        String xpath = String.format("//input[@id='newItemName%s']", dict_code);
        return By.xpath(xpath);
    }

    public void addNothing(){
        // Add --> Ok --> required feedback
        By checkEmpty = By.xpath("//div[contains(text(), \"Add\")]");
        Click(addBtn, MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(checkEmpty).size(), 1);
        Click(okBtn, MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(checkEmpty).size(), 1);

        // Add --> Close
        Click(closeBtn, MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(checkEmpty).size(), 0);
    }

    // if isDry means it wouldn't really add it.
    public void addNewDict(String newDictCode, String newDictName, String newDictDescription, boolean isDry) {
        String toCheck = String.format("//td[@id='dataDictCode%s']", newDictCode);
        Click(addBtn, MAX_BROWSER_TIMEOUT_SEC);
        SendKeys(dictCodeInput, MAX_BROWSER_TIMEOUT_SEC, newDictCode);
        SendKeys(dictNameInput, MAX_BROWSER_TIMEOUT_SEC, newDictName);
        SendKeys(dictDescription, MAX_BROWSER_TIMEOUT_SEC, newDictDescription);
        if (isDry) {
            Click(closeBtn, MAX_BROWSER_TIMEOUT_SEC);
            Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 0);
        }else {
            Click(okBtn, MAX_BROWSER_TIMEOUT_SEC);
            Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 1);
        }
    }

    public void addItem(String dictCode, String newItemCode, String newItemName, boolean isAvailable) {
        String toCheck = String.format("//td[contains(text(), \"%s\")]", newItemCode);
        waitToPresent(moreBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(moreBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(itemAddBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(itemAddBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(statusDropDown, MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(statusAvailable, MAX_BROWSER_TIMEOUT_SEC);
        if(isAvailable) {
            Click(statusAvailable, MAX_BROWSER_TIMEOUT_SEC);
        }else {
            Click(statusUnavailable, MAX_BROWSER_TIMEOUT_SEC);
        }
        SendKeys(itemCodeInput(dictCode), MAX_BROWSER_TIMEOUT_SEC, newItemCode);
        SendKeys(itemNameInput(dictCode), MAX_BROWSER_TIMEOUT_SEC, newItemName);
        Click(addActionBtn, MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 1);
        Click(okBtn, MAX_BROWSER_TIMEOUT_SEC);
        // check if add successful
        Click(moreBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(configBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 1);
        Click(okBtn, MAX_BROWSER_TIMEOUT_SEC);
    }

    public void editDict(String dictCode, String newPostfix) {
        String toCheck = String.format("//td[@id='dataDictCode%s%s']", dictCode, newPostfix);
        String checkPrivate = String.format("//td[contains(text(), \"%s_PRIVATE\")]", dictCode);
        String checkTeam = String.format("//td[contains(text(), \"%s_TEAM\")]", dictCode);
        String checkPublic = String.format("//td[contains(text(), \"%s_PUBLIC\")]", dictCode);

        waitToPresent(editBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(editBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        SendKeys(dictCodeInput, MAX_BROWSER_TIMEOUT_SEC, newPostfix);
        Click(okBtn, MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 1);
        // after edit
        Click(moreBtn(dictCode.concat(newPostfix)), MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(configBtn(dictCode.concat(newPostfix)), MAX_BROWSER_TIMEOUT_SEC);
        Click(configBtn(dictCode.concat(newPostfix)), MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(By.xpath(checkPrivate)).size(), 1);
        Assert.assertEquals(driver.findElements(By.xpath(checkTeam)).size(), 1);
        Assert.assertEquals(driver.findElements(By.xpath(checkPublic)).size(), 1);
        Click(okBtn, MAX_BROWSER_TIMEOUT_SEC);
    }

    public void deleteDict(String dictCode) {
        String toCheck = String.format("//td[@id='dataDictCode%s']", dictCode);
        Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 1);
        waitToPresent(moreBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(moreBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        Click(deleteBtn(dictCode), MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(okDeleteBtn, MAX_BROWSER_TIMEOUT_SEC);
        Click(okDeleteBtn, MAX_BROWSER_TIMEOUT_SEC);
        Assert.assertEquals(driver.findElements(By.xpath(toCheck)).size(), 0);
    }

}
