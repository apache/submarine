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


public class HistoryFileUtils {
  public static String generateFileName(JobMetadata metadata) {
    StringBuilder sb = new StringBuilder();
    sb.append(metadata.getId());
    sb.append("-");
    sb.append(metadata.getStarted());
    sb.append("-");
    if (metadata.getCompleted() != -1L) {
      sb.append(metadata.getCompleted());
      sb.append("-");
    }
    sb.append(metadata.getUser());
    if (!metadata.getStatus().isEmpty()) {
      sb.append("-");
      sb.append(metadata.getStatus());
      sb.append("." + Constants.HISTFILE_SUFFIX);
      return sb.toString();
    }
    sb.append("." + Constants.HISTFILE_SUFFIX);
    sb.append("." + Constants.INPROGRESS);
    return sb.toString();
  }

  private HistoryFileUtils() { }
}
