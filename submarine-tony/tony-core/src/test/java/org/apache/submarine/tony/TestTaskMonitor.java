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

import org.apache.submarine.tony.rpc.MetricsRpc;
import org.apache.submarine.tony.rpc.impl.MetricsWritable;
import org.apache.hadoop.conf.Configuration;
import org.mockito.Mock;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;


public class TestTaskMonitor {
  @Mock
  Configuration yarnConf = mock(Configuration.class);

  @Mock
  Configuration tonyConf = mock(Configuration.class);

  @Mock
  MetricsRpc metricsRpcClient;

  @Mock
  private TaskMonitor taskMonitor = mock(TaskMonitor.class);

  @BeforeTest
  public void setupTaskMonitor() {
    when(yarnConf.get(TonyConfigurationKeys.GPU_PATH_TO_EXEC, TonyConfigurationKeys.DEFAULT_GPU_PATH_TO_EXEC))
        .thenReturn(TonyConfigurationKeys.DEFAULT_GPU_PATH_TO_EXEC);
    when(tonyConf.getInt(TonyConfigurationKeys.getResourceKey("worker", "gpus"), 0))
        .thenReturn(1);
    taskMonitor = new TaskMonitor("worker", 0, yarnConf, tonyConf, metricsRpcClient);
    taskMonitor.initMetrics();
  }

  @Test
  public void testSetAvgMetrics() {
    taskMonitor.setAvgMetrics(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX, 0.1);
    taskMonitor.numRefreshes++;
    taskMonitor.setAvgMetrics(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX, 0.3);
    taskMonitor.numRefreshes++;
    taskMonitor.setAvgMetrics(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX, 0.5);
    MetricsWritable metrics = taskMonitor.getMetrics();
    Assert.assertEquals(metrics.getMetric(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX).getValue(), 0.3);
  }

  @Test
  public void testSetMaxMetrics() {
    taskMonitor.setMaxMetrics(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX, 0.1);
    taskMonitor.numRefreshes++;
    taskMonitor.setMaxMetrics(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX, 0.4);
    taskMonitor.numRefreshes++;
    taskMonitor.setMaxMetrics(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX, 0.2);
    MetricsWritable metrics = taskMonitor.getMetrics();
    Assert.assertEquals(metrics.getMetric(TaskMonitor.AVG_GPU_FB_MEMORY_USAGE_INDEX).getValue(), 0.4);
  }
}
