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

    public By addBtn = By.cssSelector("form > nz-form-item:nth-child(3) > nz-form-control > div > span > button.ant-btn.ant-btn-default");

    public By okBtn = By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-primary']");

    public By okDeleteBtn = By.xpath("//span[text()='Ok']/ancestor::button[@ng-reflect-nz-type='primary']");

    public By closeBtn = By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-default']");

    public By dictCodeInput = By.xpath("//input[@id='inputNewDictCode']");

    public By dictNameInput = By.xpath("//input[@id='inputNewDictName']");

    public By dictDescription = By.xpath("//input[@id='inputNewDictDescription']");

    // for configuration jump out modal
    public By statusDropDown = By.xpath("//span[@class='ant-cascader-picker-label']");

    public By statusAvailable = By.xpath("//li[@title='available']");

    public By statusUnavailable = By.xpath("//li[@title='unavailable']");

    public By addActionBtn = By.xpath("//button[@class='ant-btn ng-star-inserted ant-btn-default ant-btn-sm']");

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

}
