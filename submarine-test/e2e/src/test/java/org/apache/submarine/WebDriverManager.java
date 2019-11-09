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
import org.openqa.selenium.firefox.FirefoxDriver;
import org.rauschig.jarchivelib.Archiver;
import org.rauschig.jarchivelib.ArchiverFactory;
import java.io.File;
import java.net.URL;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WebDriverManager {
    private static Logger LOG = LoggerFactory.getLogger(WebDriverManager.class);
    private static String GECKODRIVER_VERSION = "0.25.0";
    private static String downLoadsDir = "";
    private static CommandExecutor cmdExec = new CommandExecutor();

    public static WebDriver getWebDriver() {
        WebDriver driver = null;
        // firefox webdriver
        if(driver == null) { 
            try {
                // download GeckoDriver
                downLoadsDir = System.getProperty("user.dir");
                String tempPath = downLoadsDir + "/Driver/";
                downloadGeckoDriver(tempPath);
                if(SystemUtils.IS_OS_MAC_OSX) {
                    String command = "chmod +x " + tempPath + "geckodriver"; 
                    cmdExec.executeCommandLocalHost(command);
                }
                System.setProperty("webdriver.gecko.driver", tempPath + "geckodriver");
                // TODO(Kai-Hsun Chen): set firefox preference (refer to WebDriverManager.java:74 ~ 94 in Zeppelin)
                // Initialize firefox WebDriver
                String firefoxVersion = getFirefoxVersion();
                LOG.info("Firefox version " + firefoxVersion + " detected");
                driver = new FirefoxDriver();
            } catch (Exception e) {
                LOG.info("Exception in WebDriverManager while FireFox Driver");
            }
        }

        String url = "http://127.0.0.1:32777";
        driver.get(url);
        return driver;
    }
    // TODO(Kai-Hsun Chen): need to set the path of geckodriver 
    public static void downloadGeckoDriver(String tempPath) {
        String geckoDriverUrlString = "https://github.com/mozilla/geckodriver/releases/download/v" + GECKODRIVER_VERSION + "/geckodriver-v" + GECKODRIVER_VERSION + "-";
        LOG.info("Gecko version: v" + GECKODRIVER_VERSION + ", will be downloaded to " + tempPath);
        try {
            if (SystemUtils.IS_OS_WINDOWS) {
                if (System.getProperty("sun.arch.data.model").equals("64")) {
                  geckoDriverUrlString += "win64.zip";
                } else {
                  geckoDriverUrlString += "win32.zip";
                }
            } else if (SystemUtils.IS_OS_LINUX) {
                if (System.getProperty("sun.arch.data.model").equals("64")) {
                  geckoDriverUrlString += "linux64.tar.gz";
                } else {
                  geckoDriverUrlString += "linux32.tar.gz";
                }
            } else if (SystemUtils.IS_OS_MAC_OSX) {
                geckoDriverUrlString += "macos.tar.gz";
            }
            
            File geckoDriver = new File(tempPath + "geckodriver");
            File geckoDriverZip = new File(tempPath + "geckodriver.tar");
            File geckoDriverDir = new File(tempPath);
            URL geckoDriverUrl = new URL(geckoDriverUrlString);
            if (!geckoDriver.exists()) {
              FileUtils.copyURLToFile(geckoDriverUrl, geckoDriverZip);
              if (SystemUtils.IS_OS_WINDOWS) {
                Archiver archiver = ArchiverFactory.createArchiver("zip");
                archiver.extract(geckoDriverZip, geckoDriverDir);
              } else {
                Archiver archiver = ArchiverFactory.createArchiver("tar", "gz");
                archiver.extract(geckoDriverZip, geckoDriverDir);
              }
            } else {
                LOG.info("Gecko version: v" + GECKODRIVER_VERSION + " has already existed in path " + tempPath);
                return;
            }
        } catch (Exception e) {
            LOG.info("[FAIL] Download of Gecko version: v" + GECKODRIVER_VERSION + ", falied in path " + tempPath);
            return;
        }
        LOG.info("[SUCCESS] Download of Gecko version: " + GECKODRIVER_VERSION); 
    }
    // TODO(Kai-Hsun Chen): need to be tested on Windows, MacOS, and Linux
    public static String getFirefoxVersion() {
        String firefoxVersionCmd = "firefox -v";
        String version = "";
        if (System.getProperty("os.name").startsWith("Mac OS")) {
            firefoxVersionCmd = "/Applications/Firefox.app/Contents/MacOS/" + firefoxVersionCmd;
        }
        try {
            version = cmdExec.executeCommandLocalHost(firefoxVersionCmd).toString(); 
        } catch (Exception e) {
            LOG.info("Exception in WebDriverManager while getFirefoxVersion");
        }
        return version;            
    }
}
