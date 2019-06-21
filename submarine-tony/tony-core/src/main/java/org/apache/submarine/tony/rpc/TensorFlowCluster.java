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

import org.apache.hadoop.ipc.VersionedProtocol;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.security.client.ClientToAMTokenSelector;
import org.apache.hadoop.security.token.TokenInfo;
import org.apache.hadoop.ipc.ProtocolInfo;

import java.io.IOException;

@TokenInfo(ClientToAMTokenSelector.class)
@ProtocolInfo(
    protocolName = "org.apache.submarine.tony.rpc.TensorFlowCluster",
    protocolVersion = 1)
public interface TensorFlowCluster extends VersionedProtocol {
  long versionID = 1L;

  GetTaskInfosResponse getTaskInfos(GetTaskInfosRequest request) throws IOException, YarnException;

  GetClusterSpecResponse getClusterSpec(GetClusterSpecRequest request)
      throws YarnException, IOException;

  RegisterWorkerSpecResponse registerWorkerSpec(RegisterWorkerSpecRequest request)
      throws YarnException, IOException;
  RegisterTensorBoardUrlResponse registerTensorBoardUrl(RegisterTensorBoardUrlRequest request)
      throws Exception;
  RegisterExecutionResultResponse registerExecutionResult(RegisterExecutionResultRequest request)
      throws Exception;
  Empty finishApplication(Empty request) throws YarnException, IOException;

  HeartbeatResponse taskExecutorHeartbeat(HeartbeatRequest request) throws YarnException, IOException;

}
