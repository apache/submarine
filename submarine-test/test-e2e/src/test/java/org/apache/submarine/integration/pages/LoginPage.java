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

import org.openqa.selenium.By;
import org.apache.submarine.AbstractSubmarineIT;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.PageFactory;
import org.openqa.selenium.support.pagefactory.AjaxElementLocatorFactory;


public class LoginPage extends AbstractSubmarineIT{

    private By userNameInput = By.cssSelector("input[ng-reflect-name='userName']");

    private By passwordInput = By.cssSelector("input[ng-reflect-name='password']");

    private By signInButton = By.xpath("//span[text()='Sign In']/parent::button");

    private By userNameSignInWarning = By.xpath("//div[contains(text(), \"Please input your username!\")]");

    private By passwordSignInWarning = By.xpath("//div[contains(text(), \"Please input your Password!\")]");

    private By warningText = By.xpath("//div[contains(text(), \"Username and password are incorrect,\")]");

    public LoginPage(WebDriver driver) {

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

    public void Login() {
        String username = "admin";
        String password = "admin";
        waitToPresent(userNameInput, MAX_BROWSER_TIMEOUT_SEC).sendKeys(username);
        waitToPresent(passwordInput, MAX_BROWSER_TIMEOUT_SEC).sendKeys(password);
        Click(signInButton, MAX_BROWSER_TIMEOUT_SEC);
        waitToPresent(By.cssSelector("a[routerlink='/workbench/experiment']"), MAX_BROWSER_TIMEOUT_SEC);
    }
}
