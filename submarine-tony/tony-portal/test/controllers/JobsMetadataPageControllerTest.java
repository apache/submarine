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

import org.apache.submarine.tony.TonyConfigurationKeys;
import org.junit.Test;
import play.Application;
import play.inject.guice.GuiceApplicationBuilder;
import play.mvc.Http;
import play.mvc.Result;
import play.test.WithApplication;

import static org.junit.Assert.assertEquals;
import static play.mvc.Http.Status.OK;
import static play.test.Helpers.GET;
import static play.test.Helpers.route;


public class JobsMetadataPageControllerTest extends WithApplication {

  @Override
  protected Application provideApplication() {
    Application fakeApp =
        new GuiceApplicationBuilder().configure(TonyConfigurationKeys.TONY_HISTORY_LOCATION, "/dummy/")
            .configure(TonyConfigurationKeys.TONY_HISTORY_INTERMEDIATE, "/dummy/intermediate")
            .configure(TonyConfigurationKeys.TONY_HISTORY_FINISHED, "/dummy/finished")
            .build();
    return fakeApp;
  }

  @Test
  public void testIndex() {
    Http.RequestBuilder request = new Http.RequestBuilder().method(GET).uri("/");

    Result result = route(app, request);
    assertEquals(OK, result.status());
  }
}
