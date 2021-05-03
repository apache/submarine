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

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.SystemUtils;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.firefox.FirefoxBinary;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxDriverLogLevel;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.firefox.FirefoxProfile;
import org.openqa.selenium.firefox.GeckoDriverService;
import org.rauschig.jarchivelib.Archiver;
import org.rauschig.jarchivelib.ArchiverFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

public class FirefoxWebDriverProvider implements WebDriverProvider {

  protected final static Logger LOG = LoggerFactory.getLogger(AbstractSubmarineIT.class);
  private static String GECKODRIVER_VERSION = "0.25.0";
  public final static String geckoDriverName = "geckodriver";

  @Override
  public String getWebDriverVersion() {
    return GECKODRIVER_VERSION;
  }

  @Override
  public String downloadWebDriver(String webDriverVersion) {
    String downLoadsDir = FileUtils.getTempDirectory().toString();
    String tempPath = downLoadsDir + "/firefox/";
    String geekoDriverUrlString =
        "https://github.com/mozilla/geckodriver/releases/download/v" + webDriverVersion
            + "/geckodriver-v" + webDriverVersion + "-";

    LOG.info("Geeko version: " + webDriverVersion + ", will be downloaded to " + tempPath);
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
          LOG.info("Get geckoDriver:" + geekoDriver.getAbsolutePath());
        }
      }

    } catch (IOException e) {
      LOG.error("Download of Gecko version: " + webDriverVersion +
          ", failed in path " + tempPath);
    }
    LOG.info("Download the firefox Gecko driver to " + tempPath +
        " successfully.");
    return tempPath + geckoDriverName;
  }

  @Override
  public WebDriver createWebDriver(String webDriverPath, String downloadPath) {
    FirefoxBinary ffox = new FirefoxBinary();
    if ("true".equals(System.getenv("TRAVIS"))) {
      // xvfb is supposed to run with DISPLAY 99
      ffox.setEnvironmentProperty("DISPLAY", ":99");
    }
    ffox.addCommandLineOptions("--headless");

    FirefoxProfile profile = new FirefoxProfile();
    profile.setPreference("browser.download.folderList", 2);
    profile.setPreference("browser.download.dir", downloadPath);
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

    System.setProperty(
        GeckoDriverService.GECKO_DRIVER_EXE_PROPERTY, webDriverPath);
    System.setProperty(
        FirefoxDriver.SystemProperty.DRIVER_USE_MARIONETTE, "true");

    FirefoxOptions firefoxOptions = new FirefoxOptions();
    firefoxOptions.setBinary(ffox);
    firefoxOptions.setProfile(profile);
    firefoxOptions.setLogLevel(FirefoxDriverLogLevel.TRACE);

    return new FirefoxDriver(firefoxOptions);
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
}
