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
import org.apache.submarine.tony.rpc.TensorFlowCluster;
import org.apache.hadoop.security.authorize.PolicyProvider;
import org.apache.hadoop.security.authorize.Service;

/**
 * PolicyProvider for Client to AM protocol.
 **/
public class TonyPolicyProvider extends PolicyProvider {
  @Override
  public Service[] getServices() {
    return new Service[]{
        new Service("tony.cluster", TensorFlowCluster.class),
        new Service("tony.metrics", MetricsRpc.class)
    };
  }
}
