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
import org.apache.commons.codec.binary.Base64;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.Keys;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.FluentWait;
import org.openqa.selenium.support.ui.Wait;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;

abstract public class AbstractSubmarineIT {
  protected static WebDriver driver;

  protected final static Logger LOG = LoggerFactory.getLogger(AbstractSubmarineIT.class);
  protected static final long MIN_IMPLICIT_WAIT = 5;
  protected static final long MAX_IMPLICIT_WAIT = 30;
  protected static final long MAX_BROWSER_TIMEOUT_SEC = 60;
  protected static final long MAX_PARAGRAPH_TIMEOUT_SEC = 120;

  protected void setTextOfParagraph(int paragraphNo, String text) {
    String editorId = driver.findElement(By.xpath(getParagraphXPath(paragraphNo) + "//div[contains(@class, 'editor')]")).getAttribute("id");
    if (driver instanceof JavascriptExecutor) {
      ((JavascriptExecutor) driver).executeScript("ace.edit('" + editorId + "'). setValue('" + text + "')");
    } else {
      throw new IllegalStateException("This driver does not support JavaScript!");
    }
  }

  protected void runParagraph(int paragraphNo) {
    By by = By.xpath(getParagraphXPath(paragraphNo) + "//span[@class='icon-control-play']");
    pollingWait(by, 5);
    driver.findElement(by).click();
  }


  protected String getParagraphXPath(int paragraphNo) {
    return "(//div[@ng-controller=\"ParagraphCtrl\"])[" + paragraphNo + "]";
  }

  protected String getNoteFormsXPath() {
    return "(//div[@id='noteForms'])";
  }

  protected boolean waitForParagraph(final int paragraphNo, final String state) {
    By locator = By.xpath(getParagraphXPath(paragraphNo)
        + "//div[contains(@class, 'control')]//span[2][contains(.,'" + state + "')]");
    WebElement element = pollingWait(locator, MAX_PARAGRAPH_TIMEOUT_SEC);
    return element.isDisplayed();
  }

  protected String getParagraphStatus(final int paragraphNo) {
    By locator = By.xpath(getParagraphXPath(paragraphNo)
        + "//div[contains(@class, 'control')]/span[2]");

    return driver.findElement(locator).getText();
  }

  protected boolean waitForText(final String txt, final By locator) {
    try {
      WebElement element = pollingWait(locator, MAX_BROWSER_TIMEOUT_SEC);
      return txt.equals(element.getText());
    } catch (TimeoutException e) {
      return false;
    }
  }

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

  protected String getURL(final String defaultURL, final int defaultPort) {
    String URL;
    if (System.getProperty("SUBMARINE_WORKBENCH_URL") == null) {
      URL = defaultURL;
    } else {
      URL = System.getProperty("SUBMARINE_WORKBENCH_URL"); 
    }

    URL = URL.concat(":");

    if (System.getProperty("SUBMARINE_WORKBENCH_PORT") == null) {
      URL = URL.concat(String.valueOf(defaultPort));      
    } else {
      String port = System.getProperty("SUBMARINE_WORKBENCH_PORT");
      URL = URL.concat(String.valueOf(port));
    }
    return URL;
  }

  protected WebElement buttonCheck(final By locator, final long timeWait) {
    Wait<WebDriver> wait = new FluentWait<>(driver)
        .withTimeout(timeWait, TimeUnit.SECONDS)
        .pollingEvery(1, TimeUnit.SECONDS)
        .ignoring(NoSuchElementException.class);
    return wait.until(ExpectedConditions.elementToBeClickable(locator));
  }

  protected void takeScreenShot(final String path) {
    File scrFile1 = ((TakesScreenshot)driver).getScreenshotAs(OutputType.FILE);
    try {
      FileUtils.copyFile(scrFile1, new File(path));
    } catch (java.io.IOException e) {
      e.fillInStackTrace();
    }
  }

  protected void createNewNote() {
    clickAndWait(By.xpath("//div[contains(@class, \"col-md-4\")]/div/h5/a[contains(.,'Create new" +
        " note')]"));

    WebDriverWait block = new WebDriverWait(driver, MAX_BROWSER_TIMEOUT_SEC);
    block.until(ExpectedConditions.visibilityOfElementLocated(By.id("noteCreateModal")));
    clickAndWait(By.id("createNoteButton"));
    block.until(ExpectedConditions.invisibilityOfElementLocated(By.id("createNoteButton")));
  }

  protected void deleteTestNotebook(final WebDriver driver) {
    WebDriverWait block = new WebDriverWait(driver, MAX_BROWSER_TIMEOUT_SEC);
    driver.findElement(By.xpath(".//*[@id='main']//button[@ng-click='moveNoteToTrash(note.id)']"))
        .sendKeys(Keys.ENTER);
    block.until(ExpectedConditions.visibilityOfElementLocated(By.xpath(".//*[@id='main']//button[@ng-click='moveNoteToTrash(note.id)']")));
    driver.findElement(By.xpath("//div[@class='modal-dialog'][contains(.,'This note will be moved to trash')]" +
        "//div[@class='modal-footer']//button[contains(.,'OK')]")).click();
    SubmarineITUtils.sleep(100, false);
  }

  protected void deleteTrashNotebook(final WebDriver driver) {
    WebDriverWait block = new WebDriverWait(driver, MAX_BROWSER_TIMEOUT_SEC);
    driver.findElement(By.xpath(".//*[@id='main']//button[@ng-click='removeNote(note.id)']"))
        .sendKeys(Keys.ENTER);
    block.until(ExpectedConditions.visibilityOfElementLocated(By.xpath(".//*[@id='main']//button[@ng-click='removeNote(note.id)']")));
    driver.findElement(By.xpath("//div[@class='modal-dialog'][contains(.,'This cannot be undone. Are you sure?')]" +
        "//div[@class='modal-footer']//button[contains(.,'OK')]")).click();
    SubmarineITUtils.sleep(100, false);
  }

  protected void clickAndWait(final By locator) {
    pollingWait(locator, MAX_IMPLICIT_WAIT).click();
    SubmarineITUtils.sleep(1000, false);
  }

  protected void handleException(String message, Exception e) throws Exception {
    LOG.error(message, e);
    File scrFile = ((TakesScreenshot) driver).getScreenshotAs(OutputType.FILE);
    LOG.error("ScreenShot::\ndata:image/png;base64," + new String(Base64.encodeBase64(FileUtils.readFileToByteArray(scrFile))));
    throw e;
  }

  // printSubmarineLog function will help you see submarine-dist/target/../../log/submarine.logs context
  public static void printSubmarineLog() {
    try {
      File directory = new File(".");
      String submarineDistModulePath = directory.getCanonicalPath() + "/../../submarine-dist/";
      LOG.info("submarine-dist module path = {}", submarineDistModulePath);

      File filePath = new File(submarineDistModulePath);
      String logPath = "";
      logPath = findTargetFile(filePath, "submarine.log");
      File fLog = new File(logPath);
      if (fLog.exists()) {
        LOG.info("Print Submarine log file:{}\n================= logs/submarine.log BEGIN =================", logPath);
        String line;
        StringBuffer sbContext = new StringBuffer();
        InputStream is = new FileInputStream(logPath);
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        line = reader.readLine();
        while (line != null) {
          sbContext.append(line + "\n");
          line = reader.readLine();
        }
        sbContext.append("================= logs/submarine.log END =================\n");
        LOG.info(sbContext.toString());
        reader.close();
        is.close();
      } else {
        LOG.error("Submarine log file not exist!");
      }
    } catch (IOException e) {
      LOG.error(e.getLocalizedMessage(), e);
    }
  }

  private static String findTargetFile(File dir, String targetFile) {
    File[] files = dir.listFiles();
    for (File file : files) {
      if (file.isDirectory()) {
        String targetPath = findTargetFile(file, targetFile);
        if (!targetPath.isEmpty()) {
          return targetPath;
        }
      } else {
        if (StringUtils.equals(file.getName(), targetFile)) {
          LOG.info("file : " + file.getPath());
          return file.getPath();
        }
      }
    }
    return "";
  }

  // listTargetDirFiles function will help you see some files in submarine-dist module directory
  public static void listTargetDirFiles(File dir, String targetDir) {
    LOG.info("dir:{}, targetDir:{}", dir.getName(), targetDir);
    File[] files = dir.listFiles();
    for (File file : files) {
      if (file.isDirectory()) {
        listTargetDirFiles(file, targetDir);
      } else {
        if (StringUtils.equals(dir.getName(), targetDir))
          LOG.info("file : " + file.getName());
      }
    }
  }
}
