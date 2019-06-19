/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package org.apache.submarine.tony.rpc;

import org.apache.hadoop.ipc.ProtocolInfo;
import org.apache.submarine.tony.rpc.proto.TensorFlowCluster.TensorFlowClusterService;

@ProtocolInfo(
  protocolName = "org.apache.submarine.tony.rpc.TensorFlowCluster",
  protocolVersion = 1)
public interface TensorFlowClusterPB extends TensorFlowClusterService.BlockingInterface {
}
