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
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.rauschig.jarchivelib.Archiver;
import org.rauschig.jarchivelib.ArchiverFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public class ChromeWebDriverProvider implements WebDriverProvider {

  protected final static Logger LOG = LoggerFactory.getLogger(AbstractSubmarineIT.class);
  public final static String chromeDriverName = "chromedriver";

  @Override
  public String getWebDriverVersion() {
    String chromeVersion = getChromeVersion();
    String chromeDriverVersion = getChromeDriverVersion(chromeVersion);
    return chromeDriverVersion;
  }

  @Override
  public String downloadWebDriver(String webDriverVersion) {
    String downLoadsDir = FileUtils.getTempDirectory().toString();
    String tempPath = downLoadsDir + "/chrome/";
    String chromeDriverUrlString =
        "https://chromedriver.storage.googleapis.com/" + webDriverVersion + "/";

    LOG.info("Chrome driver version: " + webDriverVersion +
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
      LOG.error("Download of chromeDriver version: " + webDriverVersion + ", falied in path " + tempPath);
    }
    LOG.info("Download the chromeDriver to " + tempPath +" successfully.");
    return tempPath + chromeDriverName;
  }

  @Override
  public WebDriver createWebDriver(String webDriverPath) {
    System.setProperty("webdriver.chrome.driver", webDriverPath);
    ChromeOptions chromeOptions = new ChromeOptions();
    // chromeOptions.addArguments("--headless");
    return new ChromeDriver(chromeOptions);
  }

  public String getChromeVersion() {
    try {
      String versionCmd = "google-chrome --version";
      if (System.getProperty("os.name").startsWith("Mac OS")) {
        versionCmd = "/Applications/Google\\ Chrome.app/Contents/MacOS/google\\ chrome --version";
      }
      String versionString = (String) CommandExecutor
          .executeCommandLocalHost(versionCmd, false, ProcessData.Types_Of_Data.OUTPUT);

      LOG.info("The version of chrome is " + versionString);
      return versionString.replaceAll("Google Chrome", "").trim();
    } catch (Exception e) {
      LOG.error("Exception in WebDriverManager while getWebDriver ", e);
      return "";
    }
  }

  public String getChromeDriverVersion(String chromeVersion) {
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

    LOG.info("The required chrome driver version is " + chromeDriverVersion);
    return chromeDriverVersion;
  }
}
