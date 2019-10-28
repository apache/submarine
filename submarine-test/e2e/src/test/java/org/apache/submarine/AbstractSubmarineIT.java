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

package org.apache.submarine;

import com.google.common.base.Function;
import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.FluentWait;
import org.openqa.selenium.support.ui.Wait;

import java.util.concurrent.TimeUnit;

abstract public class AbstractSubmarineIT{
    protected static WebDriver driver;

    //Need to add logger
    protected static final long MIN_IMPLICIT_WAIT = 5;
    protected static final long MAX_IMPLICIT_WAIT = 30;
    protected static final long MAX_BROWSER_TIMEOUT_SEC = 30;
    protected static final long MAX_PARAGRAPH_TIMEOUT_SEC = 120;

    protected WebElement pollingWait(final By locator, final long timeWait) {
        Wait<WebDriver> wait = new FluentWait<>(driver)
            .withTimeout(timeWait, TimeUnit.SECONDS)
            .pollingEvery(1, TimeUnit.SECONDS)
            .ignoring(NoSuchElementException.class);
    
        return wait.until(new Function<WebDriver, WebElement>() {
            public WebElement apply(WebDriver driver) {
                return driver.findElement(locator);
            }
        });
    }

    protected void clickAndWait(final By locator) {
        pollingWait(locator, MAX_IMPLICIT_WAIT).click();
        sleep(1000, false);
    }

    public static void sleep(long millis, boolean logOutput) {
        if (logOutput) {
          System.out.println("Starting sleeping for " + (millis / 1000) + " seconds...");
          System.out.println("Caller: " + Thread.currentThread().getStackTrace()[2]);
        }
        try {
          Thread.sleep(millis);
        } catch (InterruptedException e) {
            System.out.println("Exception in WebDriverManager while getWebDriver ");
        }
        if (logOutput) {
          System.out.println("Finished.");
        }
      }

}
