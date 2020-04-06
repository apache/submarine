package org.apache.submarine.integration;

import org.apache.submarine.AbstractSubmarineIT;
import org.apache.submarine.WebDriverManager;
import org.apache.submarine.SubmarineITUtils;
import org.openqa.selenium.By;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.Assert;

public class teamIT extends AbstractSubmarineIT {

  public final static Logger LOG = LoggerFactory.getLogger(teamIT.class);

  @BeforeClass
  public static void startUp(){
    LOG.info("[Testcase]: teamIT");
    driver =  WebDriverManager.getWebDriver();
  }

  @AfterClass
  public static void tearDown(){
    driver.quit();
  }

  @Test
  public void teamTest() throws Exception {
    // Login
    LOG.info("Login");
    pollingWait(By.cssSelector("input[ng-reflect-name='userName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    pollingWait(By.cssSelector("input[ng-reflect-name='password']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("admin");
    clickAndWait(By.cssSelector("button[class='login-form-button ant-btn ant-btn-primary']"));
    pollingWait(By.cssSelector("a[routerlink='/workbench/dashboard']"), MAX_BROWSER_TIMEOUT_SEC);

    // Routing to workspace
    pollingWait(By.xpath("//span[contains(text(), \"Workspace\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(driver.getCurrentUrl(), "http://localhost:8080/workbench/workspace");

    //Test team part
    pollingWait(By.xpath("//li[contains(text(), \"Team\")]"), MAX_BROWSER_TIMEOUT_SEC).click();
    Assert.assertEquals(pollingWait(By.xpath("//div[@id='teamDiv']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);

    clickAndWait(By.cssSelector("button[id='createTeamBtn']"));
    pollingWait(By.xpath("//input[@id='inputNewTeamName']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("NewTeamTest");
    pollingWait(By.xpath("//textarea[@id='textareaNewTeamDescription']"), MAX_BROWSER_TIMEOUT_SEC).sendKeys("NewTeamTest");
    clickAndWait(By.cssSelector("button[id='summitCreateTeam']"));
    Assert.assertEquals(pollingWait(By.xpath("//button[@id='createTeamBtn']"), MAX_BROWSER_TIMEOUT_SEC).isDisplayed(), true);

    Thread.sleep(1500);
  }
}