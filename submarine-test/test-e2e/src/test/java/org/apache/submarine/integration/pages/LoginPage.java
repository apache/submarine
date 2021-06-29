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
import org.apache.submarine.AbstractSubmarineIT;
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

public class LoginPage extends AbstractSubmarineIT{

    @FindBy(css = "input[ng-reflect-name='userName']")
    private By userNameInput;

    @FindBy(css = "input[ng-reflect-name='password']")
    private By passwordInput;

    @FindBy(xpath = "//span[text()='Sign In']/parent::button")
    private By signInButton;

    @FindBy(xpath = "//div[contains(text(), \"Please input your username!\")]")
    private By userNameSignInWarning;

    @FindBy(xpath = "//div[contains(text(), \"Please input your Password!\")]")
    private By passwordSignInWarning;

    @FindBy(xpath = "//div[contains(text(), \"Username and password are incorrect,\")]")
    private By warningText;

    private Actions action;

    public LoginPage(WebDriver driver) {
        PageFactory.initElements(new AjaxElementLocatorFactory(driver, 10), this);
        action = new Actions(driver);
    }

    public void clickSignInBtn() {
        buttonCheck(signInButton, MAX_BROWSER_TIMEOUT_SEC).click();
    }

    public void fillLoginForm(String username, String password) {
        SendKeys(userNameInput, MAX_BROWSER_TIMEOUT_SEC, username);
        SendKeys(passwordInput, MAX_BROWSER_TIMEOUT_SEC, password);
    }

    public void waitForWarningPresent() {
        waitToPresent(warningText, MAX_BROWSER_TIMEOUT_SEC);
    }

    public By getUserNameSignInWarning() {
        return userNameSignInWarning;
    }

    public By getPasswordSignInWarning() {
        return passwordSignInWarning;
    }
}
