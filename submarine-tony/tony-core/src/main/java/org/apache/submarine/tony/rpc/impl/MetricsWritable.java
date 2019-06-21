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
import org.apache.submarine.tony.rpc.MetricWritable;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.hadoop.io.Writable;


/**
 * For serializing an array of {@link MetricWritable} over the wire.
 */
public class MetricsWritable implements Writable {
  private MetricWritable[] metrics;

  // Required for serialization
  public MetricsWritable() { }

  public MetricsWritable(int numMetrics) {
    this.metrics = new MetricWritable[numMetrics];
  }

  public MetricWritable getMetric(int index) {
    return metrics[index];
  }

  public void setMetric(int index, MetricWritable metric) {
    metrics[index] = metric;
  }

  public List<Metric> getMetricsAsList() {
    return Arrays.stream(metrics).map(metric -> new Metric(metric.getName(), metric.getValue()))
        .collect(Collectors.toList());
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(metrics.length);
    for (MetricWritable metric : metrics) {
      metric.write(out);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    metrics = new MetricWritable[in.readInt()];
    for (int i = 0; i < metrics.length; i++) {
      MetricWritable metric = new MetricWritable();
      metric.readFields(in);
      metrics[i] = metric;
    }
  }
}
