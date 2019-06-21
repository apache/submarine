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
package org.apache.submarine.tony.util;

import org.apache.submarine.tony.Constants;
import org.apache.submarine.tony.models.JobMetadata;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;


public class TestHistoryFileUtils {
  @Test
  public void testGenerateFileNameInProgressJob() {
    String appId = "app123";
    long started = 1L;
    String user = "user";

    JobMetadata metadata = new JobMetadata.Builder()
        .setId(appId)
        .setStarted(started)
        .setUser(user)
        .setConf(new YarnConfiguration())
        .build();
    String expectedName = "app123-1-user." + Constants.HISTFILE_SUFFIX + "." + Constants.INPROGRESS;

    assertEquals(HistoryFileUtils.generateFileName(metadata), expectedName);
  }

  @Test
  public void testGenerateFileNameFinishedJob() {
    String appId = "app123";
    long started = 1L;
    long completed = 2L;
    String user = "user";

    JobMetadata metadata = new JobMetadata.Builder()
        .setId(appId)
        .setStarted(started)
        .setCompleted(completed)
        .setUser(user)
        .setStatus(Constants.SUCCEEDED)
        .setConf(new YarnConfiguration())
        .build();
    String expectedName = "app123-1-2-user-SUCCEEDED." + Constants.HISTFILE_SUFFIX;

    assertEquals(HistoryFileUtils.generateFileName(metadata), expectedName);
  }
}
