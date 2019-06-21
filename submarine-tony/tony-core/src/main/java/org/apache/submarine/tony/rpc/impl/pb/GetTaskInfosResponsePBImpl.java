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

import org.apache.submarine.tony.rpc.TaskInfo;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos;
import org.apache.submarine.tony.util.ProtoUtils;
import org.apache.submarine.tony.rpc.GetTaskInfosResponse;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetTaskInfosResponseProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetTaskInfosResponseProtoOrBuilder;
import java.util.Set;
import java.util.stream.Collectors;


public class GetTaskInfosResponsePBImpl implements GetTaskInfosResponse {
  GetTaskInfosResponseProto proto = GetTaskInfosResponseProto.getDefaultInstance();
  GetTaskInfosResponseProto.Builder builder = null;
  private boolean viaProto = false;

  private Set<TaskInfo> _taskInfos = null;

  public GetTaskInfosResponsePBImpl() {
    builder = YarnTensorFlowClusterProtos.GetTaskInfosResponseProto.newBuilder();
  }

  public GetTaskInfosResponsePBImpl(GetTaskInfosResponseProto proto) {
    this.proto = proto;
    viaProto = true;
  }

  public YarnTensorFlowClusterProtos.GetTaskInfosResponseProto getProto() {
    mergeLocalToProto();
    proto = viaProto ? proto : builder.build();
    viaProto = true;
    return proto;
  }

  private void mergeLocalToProto() {
    if (viaProto) {
      maybeInitBuilder();
    }
    proto = builder.build();
    viaProto = true;
  }

  private void maybeInitBuilder() {
    if (viaProto || builder == null) {
      builder = GetTaskInfosResponseProto.newBuilder(proto);
    }
    viaProto = false;
  }

  @Override
  public Set<TaskInfo> getTaskInfos() {
    GetTaskInfosResponseProtoOrBuilder p = viaProto ? proto : builder;
    if (this._taskInfos != null) {
      return this._taskInfos;
    }
    return p.getTaskInfosList().stream().map(ProtoUtils::taskInfoProtoToTaskInfo).collect(Collectors.toSet());
  }

  @Override
  public void setTaskInfos(Set<TaskInfo> taskInfos) {
    maybeInitBuilder();
    this._taskInfos = taskInfos;
    builder.clearTaskInfos();
    builder.addAllTaskInfos(taskInfos.stream().map(ProtoUtils::taskInfoToTaskInfoProto)
        .collect(Collectors.toList()));
  }
}
