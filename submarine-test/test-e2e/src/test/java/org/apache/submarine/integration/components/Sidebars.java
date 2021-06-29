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

package org.apache.submarine.integration.components;

import org.openqa.selenium.By;
import org.apache.submarine.AbstractSubmarineIT;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.PageFactory;
import org.openqa.selenium.support.pagefactory.AjaxElementLocatorFactory;

public class Sidebars extends AbstractSubmarineIT{

    private String URL = getURL("http://127.0.0.1", 8080);

    private By toNoteBook = By.xpath("//span[contains(text(), \"Notebook\")]");
    private String noteBookURL = "/workbench/notebook";

    private By toExperiment = By.xpath("//span[contains(text(), \"Experiment\")]");
    private String experimentURL = "/workbench/experiment";

    private By toTemplate = By.xpath("//span[contains(text(), \"Template\")]");
    private String templateURL = "/workbench/template";

    private By toEnvironment = By.xpath("//span[contains(text(), \"Environment\")]");
    private String environmentURL = "/workbench/environment";

    private By toManager = By.xpath("//span[contains(text(), \"Manager\")]");
    private String managerURL = "/workbench/manager";

    private By toDataDict = By.xpath("//a[@href='/workbench/manager/dataDict']");
    private String dataDictURL = "/workbench/manager/dataDict";

    private By toDepartment = By.xpath("//a[@href='/workbench/manager/department']");
    private String departmentURL = "/workbench/manager/department";

    private By toUser = By.xpath("//a[@href='/workbench/manager/user']");
    private String userURL = "/workbench/manager/user";

    private Actions action;

    public void Sidebars(WebDriver driver) {
        PageFactory.initElements(new AjaxElementLocatorFactory(driver, 10), this);
        action = new Actions(driver);
    }

    public void gotoNoteBook() {
        ClickAndNavigate(toNoteBook, MAX_BROWSER_TIMEOUT_SEC, URL.concat(noteBookURL));
    }

    public void gotoExperiment() {
        ClickAndNavigate(toExperiment, MAX_BROWSER_TIMEOUT_SEC, URL.concat(experimentURL));
    }

    public void gotoTemplate() {
        ClickAndNavigate(toTemplate, MAX_BROWSER_TIMEOUT_SEC, URL.concat(templateURL));
    }

    public void gotoEnvironment() {
        ClickAndNavigate(toEnvironment, MAX_BROWSER_TIMEOUT_SEC, URL.concat(experimentURL));
    }

    public void gotoUser() {
        Click(toManager, MAX_BROWSER_TIMEOUT_SEC);
        ClickAndNavigate(toUser, MAX_BROWSER_TIMEOUT_SEC, URL.concat(userURL));
    }

    public void gotoDataDict() {
        Click(toManager, MAX_BROWSER_TIMEOUT_SEC);
        ClickAndNavigate(toDataDict, MAX_BROWSER_TIMEOUT_SEC, URL.concat(dataDictURL));
    }

    public void gotoDepartment() {
        Click(toManager, MAX_BROWSER_TIMEOUT_SEC);
        ClickAndNavigate(toDepartment, MAX_BROWSER_TIMEOUT_SEC, URL.concat(departmentURL));
    }



}
