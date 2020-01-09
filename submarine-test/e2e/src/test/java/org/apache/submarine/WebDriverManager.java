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

import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.TimeUnit;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.SystemUtils;
import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxBinary;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxDriver.SystemProperty;
import org.openqa.selenium.firefox.FirefoxDriverLogLevel;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.firefox.FirefoxProfile;
import org.openqa.selenium.firefox.GeckoDriverService;
import org.openqa.selenium.safari.SafariDriver;
import org.openqa.selenium.support.ui.ExpectedCondition;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.rauschig.jarchivelib.Archiver;
import org.rauschig.jarchivelib.ArchiverFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class WebDriverManager {

  public final static Logger LOG = LoggerFactory.getLogger(WebDriverManager.class);

  private static String downLoadsDir = "";

  private static String GECKODRIVER_VERSION = "0.25.0";

  private static String chromeDriverName = "chromedriver";

  public static WebDriver getWebDriver() {
    WebDriver driver = null;

    if (driver == null) {
      try {
        downLoadsDir = FileUtils.getTempDirectory().toString();
        String tempPath = downLoadsDir + "/chrome/";
        String chromeVersion = getChromeVersion();
        String chromeDriverVersion = getChromeDriverVersion(chromeVersion);
        downloadChromeDriver(chromeDriverVersion, tempPath);
        System.setProperty("webdriver.chrome.driver", tempPath + chromeDriverName);

        ChromeOptions chromeOptions = new ChromeOptions();
        chromeOptions.addArguments("--headless");
        driver = new ChromeDriver(chromeOptions);
      } catch (Exception e) {
        LOG.error("Exception in WebDriverManager while ChromeDriver ", e);
      }
    }

    if (driver == null) {
      try {
        FirefoxBinary ffox = new FirefoxBinary();
        if ("true".equals(System.getenv("TRAVIS"))) {
          // xvfb is supposed to run with DISPLAY 99
           ffox.setEnvironmentProperty("DISPLAY", ":99");
         }
        ffox.addCommandLineOptions("--headless");

        int firefoxVersion = WebDriverManager.getFirefoxVersion();
        LOG.info("Firefox version " + firefoxVersion + " detected");

        downLoadsDir = FileUtils.getTempDirectory().toString();

        String tempPath = downLoadsDir + "/firefox/";

        downloadGeekoDriver(firefoxVersion, tempPath);

        FirefoxProfile profile = new FirefoxProfile();
        profile.setPreference("browser.download.folderList", 2);
        profile.setPreference("browser.download.dir", downLoadsDir);
        profile.setPreference("browser.helperApps.alwaysAsk.force", false);
        profile.setPreference("browser.download.manager.showWhenStarting", false);
        profile.setPreference("browser.download.manager.showAlertOnComplete", false);
        profile.setPreference("browser.download.manager.closeWhenDone", true);
        profile.setPreference("app.update.auto", false);
        profile.setPreference("app.update.enabled", false);
        profile.setPreference("dom.max_script_run_time", 0);
        profile.setPreference("dom.max_chrome_script_run_time", 0);
        profile.setPreference("browser.helperApps.neverAsk.saveToDisk",
            "application/x-ustar,application/octet-stream,application/zip,text/csv,text/plain");
        profile.setPreference("network.proxy.type", 0);

        System.setProperty(GeckoDriverService.GECKO_DRIVER_EXE_PROPERTY, tempPath + "geckodriver");
        System.setProperty(SystemProperty.DRIVER_USE_MARIONETTE, "false");

        FirefoxOptions firefoxOptions = new FirefoxOptions();
        firefoxOptions.setBinary(ffox);
        firefoxOptions.setProfile(profile);
        firefoxOptions.setLogLevel(FirefoxDriverLogLevel.TRACE);
        driver = new FirefoxDriver(firefoxOptions);
      } catch (Exception e) {
        LOG.error("Exception in WebDriverManager while FireFox Driver ", e);
      }
    }

    if (driver == null) {
      try {
        driver = new SafariDriver();
      } catch (Exception e) {
        LOG.error("Exception in WebDriverManager while SafariDriver ", e);
      }
    }

    String url;
    if (System.getenv("url") != null) {
      url = System.getenv("url");
    } else {
      url = "http://localhost:8080";
    }

    long start = System.currentTimeMillis();
    boolean loaded = false;
    driver.manage().timeouts().implicitlyWait(AbstractSubmarineIT.MAX_IMPLICIT_WAIT,
        TimeUnit.SECONDS);
    driver.get(url);

    while (System.currentTimeMillis() - start < 60 * 1000) {
      // wait for page load
      try {
        (new WebDriverWait(driver, 30)).until(new ExpectedCondition<Boolean>() {
          @Override
          public Boolean apply(WebDriver d) {
            // return d.findElement(By.tagName("div"))
            return d.findElement(By.tagName("submarine-root"))
                .isDisplayed();
          }
        });
        loaded = true;
        break;
      } catch (TimeoutException e) {
        LOG.info("Exception in WebDriverManager while WebDriverWait ", e);
        driver.navigate().to(url);
      }
    }

    if (loaded == false) {
      fail();
    }

    driver.manage().window().maximize();
    return driver;
  }

  public static void downloadGeekoDriver(int firefoxVersion, String tempPath) {
    String geekoDriverUrlString =
        "https://github.com/mozilla/geckodriver/releases/download/v" + GECKODRIVER_VERSION
            + "/geckodriver-v" + GECKODRIVER_VERSION + "-";

    LOG.info("Geeko version: " + GECKODRIVER_VERSION + ", will be downloaded to " + tempPath);
    try {
      if (SystemUtils.IS_OS_WINDOWS) {
        if (System.getProperty("sun.arch.data.model").equals("64")) {
          geekoDriverUrlString += "win64.zip";
        } else {
          geekoDriverUrlString += "win32.zip";
        }
      } else if (SystemUtils.IS_OS_LINUX) {
        if (System.getProperty("sun.arch.data.model").equals("64")) {
          geekoDriverUrlString += "linux64.tar.gz";
        } else {
          geekoDriverUrlString += "linux32.tar.gz";
        }
      } else if (SystemUtils.IS_OS_MAC_OSX) {
        geekoDriverUrlString += "macos.tar.gz";
      }

      File geekoDriver = new File(tempPath + "geckodriver");
      File geekoDriverZip = new File(tempPath + "geckodriver.tar");
      File geekoDriverDir = new File(tempPath);
      URL geekoDriverUrl = new URL(geekoDriverUrlString);
      if (!geekoDriver.exists()) {
        FileUtils.copyURLToFile(geekoDriverUrl, geekoDriverZip);
        if (SystemUtils.IS_OS_WINDOWS) {
          Archiver archiver = ArchiverFactory.createArchiver("zip");
          archiver.extract(geekoDriverZip, geekoDriverDir);
        } else {
          Archiver archiver = ArchiverFactory.createArchiver("tar", "gz");
          archiver.extract(geekoDriverZip, geekoDriverDir);
          LOG.info("Get geekoDriver:" + geekoDriver.getAbsolutePath());
        }
      }

    } catch (IOException e) {
      LOG.error("Download of Geeko version: " + GECKODRIVER_VERSION + ", falied in path " + tempPath);
    }
    LOG.info("Download of Geeko version: " + GECKODRIVER_VERSION + ", successful");
  }

  public static void downloadChromeDriver(String chromeDriverVersion,
      String tempPath) {
    String chromeDriverUrlString =
        "https://chromedriver.storage.googleapis.com/"
            + chromeDriverVersion + "/";

    LOG.info("Chrome driver version: " + chromeDriverVersion +
        ", will be downloaded to " + tempPath);
    String chromeDriverZipName = "";
    try {
      if (SystemUtils.IS_OS_WINDOWS) {
        if (System.getProperty("sun.arch.data.model").equals("32")) {
          chromeDriverZipName = "chromedriver_win32.zip";
        }
      } else if (SystemUtils.IS_OS_LINUX) {
        if (System.getProperty("sun.arch.data.model").equals("64")) {
          chromeDriverZipName = "chromedriver_linux64.zip";
        }
      } else if (SystemUtils.IS_OS_MAC_OSX) {
        chromeDriverZipName = "chromedriver_mac64.zip";
      }
      chromeDriverUrlString += chromeDriverZipName;

      File chromeDriver = new File(tempPath + chromeDriverName);
      File chromeDriverZip = new File(tempPath + chromeDriverZipName);
      File chromeDriverDir = new File(tempPath);
      URL driverUrl = new URL(chromeDriverUrlString);
      if (!chromeDriver.exists()) {
        FileUtils.copyURLToFile(driverUrl, chromeDriverZip);
        if (SystemUtils.IS_OS_WINDOWS) {
          Archiver archiver = ArchiverFactory.createArchiver("zip");
          archiver.extract(chromeDriverZip, chromeDriverDir);
        } else {
          Archiver archiver = ArchiverFactory.createArchiver("zip");
          archiver.extract(chromeDriverZip, chromeDriverDir);
          LOG.info("Get chromeDriver:" + chromeDriver.getAbsolutePath());
        }
      }

    } catch (IOException e) {
      LOG.error("Download of chromeDriver version: " + chromeDriverVersion + ", falied in path " + tempPath);
    }
    LOG.info("Download of chromeDriver version: " + chromeDriverVersion + ", successful");
  }

  public static int getFirefoxVersion() {
    try {
      String firefoxVersionCmd = "firefox -v";
      if (System.getProperty("os.name").startsWith("Mac OS")) {
        firefoxVersionCmd = "/Applications/Firefox.app/Contents/MacOS/" + firefoxVersionCmd;
      }
      String versionString = (String) CommandExecutor
          .executeCommandLocalHost(firefoxVersionCmd, false, ProcessData.Types_Of_Data.OUTPUT);
      return Integer
          .valueOf(versionString.replaceAll("Mozilla Firefox", "").trim().substring(0, 2));
    } catch (Exception e) {
      LOG.error("Exception in WebDriverManager while getWebDriver ", e);
      return -1;
    }
  }

  public static String getChromeVersion() {
    try {
      String versionCmd = "google-chrome --version";
      if (System.getProperty("os.name").startsWith("Mac OS")) {
        versionCmd = "/Applications/Google Chrome.app/Contents/MacOS/google chrome --version";
      }
      String versionString = (String) CommandExecutor
          .executeCommandLocalHost(versionCmd, false, ProcessData.Types_Of_Data.OUTPUT);
      return versionString.replaceAll("Google Chrome", "").trim();
    } catch (Exception e) {
      LOG.error("Exception in WebDriverManager while getWebDriver ", e);
      return "";
    }
  }

  public static String getChromeDriverVersion(String chromeVersion) {
    // chromeVersion is like 75.0.3770.140.
    // Get the major version of chrome, like 75.
    String chromeMajorVersion =
        chromeVersion.substring(0, chromeVersion.indexOf("."));
    String chromeDriverIndexUrl =
        "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_"
            + chromeMajorVersion;

    // Get the chrome driver version according to the chrome.
    File chromeDriverVersionFile = new File(
        FileUtils.getTempDirectory().toString() + "/chromeDriverVersion");

    String chromeDriverVersion = "";
    try {
      FileUtils.copyURLToFile(
          new URL(chromeDriverIndexUrl), chromeDriverVersionFile);
      chromeDriverVersion = new String(
          Files.readAllBytes(chromeDriverVersionFile.toPath()),
          StandardCharsets.UTF_8);
      chromeDriverVersionFile.delete();
    } catch (Exception e) {
      LOG.error("Exception in getting chromeDriverVersion", e);
    }
    return chromeDriverVersion;
  }
}
