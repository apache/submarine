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

package org.apache.submarine.tony.rpc;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.Writable;


/**
 * For serializing a {@link org.apache.submarine.tony.events.Metric} over the wire.
 */
public class MetricWritable implements Writable {
  private String name;
  private double value;

  public MetricWritable() { }

  public MetricWritable(String name, double value) {
    this.name = name;
    this.value = value;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public double getValue() {
    return value;
  }

  public void setValue(double value) {
    this.value = value;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(name);
    out.writeDouble(value);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    name = in.readUTF();
    value = in.readDouble();
  }
}
