/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package controllers;

import com.google.common.collect.ImmutableMap;
import org.apache.submarine.tony.TonyConfigurationKeys;
import org.junit.Test;
import play.Application;
import play.test.Helpers;
import play.test.TestBrowser;
import play.test.WithBrowser;

import static org.junit.Assert.assertTrue;
import static play.test.Helpers.fakeApplication;


public class BrowserTest extends WithBrowser {

  protected Application provideApplication() {
    return fakeApplication(ImmutableMap.of(
        TonyConfigurationKeys.TONY_HISTORY_LOCATION, "/dummy/",
        TonyConfigurationKeys.TONY_HISTORY_INTERMEDIATE, "/dummy/intermediate",
        TonyConfigurationKeys.TONY_HISTORY_FINISHED, "/dummy/finished")
    );
  }

  protected TestBrowser provideBrowser(int port) {
    return Helpers.testBrowser(port);
  }

  @Test
  public void test() {
    browser.goTo("http://localhost:" + play.api.test.Helpers.testServerPort());
    assertTrue(browser.pageSource().contains("TonY Portal"));
  }
}
