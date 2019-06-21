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
package org.apache.submarine.tony;

import org.apache.hadoop.conf.Configuration;
import org.testng.annotations.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;


public class TestTaskExecutor {

  @Test(expectedExceptions = IllegalArgumentException.class)
  public void testTaskExecutorConfShouldThrowException() throws Exception {
    TaskExecutor taskExecutor = new TaskExecutor();
    Configuration tonyConf = new Configuration(false);
    tonyConf.setInt(TonyConfigurationKeys.TASK_HEARTBEAT_INTERVAL_MS, 2000);
    File confFile = new File(System.getProperty("user.dir"), Constants.TONY_FINAL_XML);
    try (OutputStream os = new FileOutputStream(confFile)) {
      tonyConf.writeXml(os);
    }
    if (!confFile.delete()) {
      throw new RuntimeException("Failed to delete conf file");
    }
    // Should throw exception since we didn't set up Task Command.
    taskExecutor.initConfigs();
  }

}
