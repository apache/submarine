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

package org.apache.submarine.tony.rpc.impl;

import org.apache.submarine.tony.events.avro.Metric;
import org.apache.submarine.tony.rpc.MetricsRpc;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.ipc.ProtocolSignature;


/**
 * Stores metrics and handles metric updates for all tasks.
 */
public class MetricsRpcServer implements MetricsRpc {
  private static final Log LOG = LogFactory.getLog(MetricsRpcServer.class);

  private Map<String, Map<Integer, MetricsWritable>> metricsMap = new HashMap<>();

  public List<Metric> getMetrics(String taskType, int taskIndex) {
    if (!metricsMap.containsKey(taskType) || !metricsMap.get(taskType).containsKey(taskIndex)) {
      LOG.warn("No metrics for " + taskType + " " + taskIndex + "!");
      return Collections.EMPTY_LIST;
    }
    return metricsMap.get(taskType).get(taskIndex).getMetricsAsList();
  }

  /**
   * Replaces the metrics stored for {@code taskType} {@code taskIndex} with {@code metrics}.
   */
  @Override
  public void updateMetrics(String taskType, int taskIndex, MetricsWritable metrics) {
    if (!metricsMap.containsKey(taskType)) {
      metricsMap.put(taskType, new HashMap<>());
    }
    metricsMap.get(taskType).put(taskIndex, metrics);
  }

  @Override
  public long getProtocolVersion(String protocol, long clientVersion) {
    return versionID;
  }

  @Override
  public ProtocolSignature getProtocolSignature(String protocol, long clientVersion, int clientMethodsHash)
      throws IOException {
    return ProtocolSignature.getProtocolSignature(this, protocol, clientVersion, clientMethodsHash);
  }
}
