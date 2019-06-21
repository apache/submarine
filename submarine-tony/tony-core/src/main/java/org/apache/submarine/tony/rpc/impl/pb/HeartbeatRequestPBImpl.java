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

package org.apache.submarine.tony.rpc.impl.pb;

import org.apache.submarine.tony.rpc.HeartbeatRequest;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.HeartbeatRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.HeartbeatRequestProtoOrBuilder;


public class HeartbeatRequestPBImpl implements HeartbeatRequest {
  HeartbeatRequestProto proto = HeartbeatRequestProto.getDefaultInstance();
  HeartbeatRequestProto.Builder builder = null;
  private boolean viaProto = false;

  private String taskId = null;

  public HeartbeatRequestPBImpl() {
    builder = HeartbeatRequestProto.newBuilder();
  }

  public HeartbeatRequestPBImpl(HeartbeatRequestProto proto) {
    this.proto = proto;
    viaProto = true;
  }

  public HeartbeatRequestProto getProto() {
    mergeLocalToProto();
    proto = viaProto ? proto : builder.build();
    viaProto = true;
    return proto;
  }

  private void mergeLocalToProto() {
    if (viaProto) {
      maybeInitBuilder();
    }
    mergeLocalToBuilder();
    proto = builder.build();
    viaProto = true;
  }

  private void mergeLocalToBuilder() {
    if (this.taskId != null) {
      builder.setTaskId(this.taskId);
    }
  }

  private void maybeInitBuilder() {
    if (viaProto || builder == null) {
      builder = HeartbeatRequestProto.newBuilder(proto);
    }
    viaProto = false;
  }

  @Override
  public String getTaskId() {
    HeartbeatRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (this.taskId != null) {
      return this.taskId;
    }
    if (!p.hasTaskId()) {
      return null;
    }
    this.taskId = p.getTaskId();
    return this.taskId;
  }

  @Override
  public void setTaskId(String taskId) {
    maybeInitBuilder();
    if (taskId == null) {
      builder.clearTaskId();
    }
    this.taskId = taskId;
  }
}
