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
import org.apache.submarine.tony.models.JobEvent;
import org.apache.submarine.tony.util.HdfsUtils;
import org.apache.submarine.tony.util.ParserUtils;
import hadoop.Requirements;
import java.util.List;
import javax.inject.Inject;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import play.mvc.Controller;
import play.mvc.Result;


public class JobEventPageController extends Controller {
  private FileSystem myFs;
  private Cache<String, List<JobEvent>> cache;
  private Path interm;
  private Path finished;

  @Inject
  public JobEventPageController(Requirements requirements, CacheWrapper cacheWrapper) {
    myFs = requirements.getFileSystem();
    cache = cacheWrapper.getEventCache();
    interm = requirements.getIntermediateDir();
    finished = requirements.getFinishedDir();
  }

  public Result index(String jobId) {
    List<JobEvent> listOfEvents;
    if (myFs == null) {
      return internalServerError("Failed to initialize file system in " + this.getClass());
    }

    // Check cache
    listOfEvents = cache.getIfPresent(jobId);
    if (listOfEvents != null) {
      return ok(views.html.event.render(listOfEvents));
    }

    // Check finished dir
    Path jobFolder = HdfsUtils.getJobDirPath(myFs, finished, jobId);
    if (jobFolder != null) {
      listOfEvents = ParserUtils.mapEventToJobEvent(ParserUtils.parseEvents(myFs, jobFolder));
      cache.put(jobId, listOfEvents);
      return ok(views.html.event.render(listOfEvents));
    }

    // Check intermediate dir
    jobFolder = HdfsUtils.getJobDirPath(myFs, interm, jobId);
    if (jobFolder != null) {
      return internalServerError("Cannot display events because job is still running");
    }

    return internalServerError("Failed to fetch events");
  }
}
