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

public class DataDictPage extends AbstractSubmarineIT {

    private By addBtn = By.cssSelector("form > nz-form-item:nth-child(3) > nz-form-control > div > span > button.ant-btn.ant-btn-default");

    private By okBtn = By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-primary']");

    private By closeBtn = By.cssSelector("button[class='ant-btn ng-star-inserted ant-btn-default']");

    private By dictCodeInput = By.xpath("//input[@id='inputNewDictCode']");

    private By dictNameInput = By.xpath("//input[@id='inputNewDictName']");

    private By dictDescription = By.xpath("//input[@id='inputNewDictDescription']");

    // buttons for PROJECT_TYPE
    private By moreBtn = By.xpath("//a[@id='dataDictMorePROJECT_TYPE']");

    private By configBtn = By.xpath("//li[@id='dataDictConfigurationPROJECT_TYPE']");

    private By configAddBtn = By.xpath("//button[@id='dataDictItemAddPROJECT_TYPE']");

    private By itemCodeInput = By.xpath("//input[@id='newItemCodePROJECT_TYPE']");

    private By itemNameInput = By.xpath("//input[@id='newItemNamePROJECT_TYPE']");

    // buttons for PROJECT_VISIBILITY and its update

    private By editBtn = By.xpath("//a[@id='dataDictEditPROJECT_VISIBILITY']");


    // for configuration jump out modal
    private By statusDropDown = By.xpath("//span[@class='ant-cascader-picker-label']");

    private By statusUnavailable = By.xpath("//li[@title='unavailable']");

    private By addItemBtn = By.xpath("//button[@class='ant-btn ng-star-inserted ant-btn-default ant-btn-sm']");

    // for edit jump out modal

}
