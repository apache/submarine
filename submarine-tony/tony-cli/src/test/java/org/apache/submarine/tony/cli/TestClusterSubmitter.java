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

package org.apache.submarine.tony.cli;

import org.apache.submarine.tony.Constants;
import org.apache.submarine.tony.TonyClient;
import org.apache.submarine.tony.TonyConfigurationKeys;
import org.testng.annotations.Test;

import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;


public class TestClusterSubmitter {
  @Test
  public void testClusterSubmitter() throws  Exception {
    TonyClient client = spy(new TonyClient());
    doReturn(0).when(client).start(); // Don't really call start() method.

    ClusterSubmitter submitter = new ClusterSubmitter(client);
    int exitCode = submitter.submit(new String[] {"--src_dir", "src"});
    assertEquals(exitCode, 0);
    assertTrue(client.getTonyConf().get(TonyConfigurationKeys.getContainerResourcesKey())
        .contains(Constants.TONY_JAR_NAME));
  }
}

