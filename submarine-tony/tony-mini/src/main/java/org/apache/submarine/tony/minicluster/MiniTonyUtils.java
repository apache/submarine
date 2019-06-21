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

package org.apache.submarine.tony.minicluster;

import java.io.IOException;
import java.io.PrintWriter;
import org.apache.hadoop.conf.Configuration;


public class MiniTonyUtils {

  /**
   * Write a Hadoop configuration to file.
   * @param conf the configuration object.
   * @param filePath the filePath we are writing the configuration to.
   * @throws IOException IO exception during writing files.
   */
  public static void saveConfigToFile(Configuration conf, String filePath) throws IOException {
    PrintWriter yarnWriter = new PrintWriter(filePath, "UTF-8");
    conf.writeXml(yarnWriter);
  }

  private MiniTonyUtils() { }

}
