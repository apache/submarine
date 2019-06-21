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
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


public class HDFSUtils {
  private static final Log LOG = LogFactory.getLog(HDFSUtils.class);

  /**
   * Copy files under src directory recursively to dst folder.
   * @param fs a hadoop file system reference
   * @param src the directory under which you want to copy files from (local disk)
   * @param dst the destination directory. (hdfs)
   * @throws IOException exception when copy files.
   */
  public static void copyDirectoryFilesToFolder(FileSystem fs, String src, String dst) throws IOException {
    Files.walk(Paths.get(src))
        .filter(Files::isRegularFile)
        .forEach(file -> {
          Path jar = new Path(file.toString());
          try {
            fs.copyFromLocalFile(jar, new Path(dst));
          } catch (IOException e) {
            LOG.error("Failed to copy directory from: " + src + " to: " + dst + " ", e);
          }
        });
  }

  private HDFSUtils() { }
}
