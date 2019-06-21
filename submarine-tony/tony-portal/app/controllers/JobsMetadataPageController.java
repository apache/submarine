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

import cache.CacheWrapper;
import com.google.common.cache.Cache;
import org.apache.submarine.tony.models.JobMetadata;
import javax.inject.Inject;
import play.mvc.Controller;
import play.mvc.Result;


public class JobsMetadataPageController extends Controller {
  private Cache<String, JobMetadata> cache;

  @Inject
  public JobsMetadataPageController(CacheWrapper cacheWrapper) {
    cache = cacheWrapper.getMetadataCache();
  }

  public Result index() {
    return ok(views.html.metadata.render(cache.asMap().values()));
  }
}
