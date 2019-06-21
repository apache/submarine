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

import org.apache.submarine.tony.rpc.RegisterExecutionResultRequest;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProto;


public class RegisterExecutionResultRequestPBImpl implements RegisterExecutionResultRequest {
  private RegisterExecutionResultRequestProto proto
      = RegisterExecutionResultRequestProto.getDefaultInstance();
  private RegisterExecutionResultRequestProto.Builder builder = null;
  private boolean viaProto = false;
  private String jobName = null;
  private String jobIndex = null;
  private String sessionId = null;

  public RegisterExecutionResultRequestPBImpl() {
    builder = RegisterExecutionResultRequestProto.newBuilder();
  }

  public RegisterExecutionResultRequestPBImpl(RegisterExecutionResultRequestProto proto) {
    this.proto = proto;
    viaProto = true;
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
    if (this.jobName != null) {
      builder.setJobName(this.jobName);
    }
    if (this.jobIndex != null) {
      builder.setJobIndex(this.jobIndex);
    }
    if (this.sessionId != null) {
      builder.setSessionId(this.sessionId);
    }
  }

  public RegisterExecutionResultRequestProto getProto() {
    mergeLocalToProto();
    proto = viaProto ? proto : builder.build();
    viaProto = true;
    return proto;
  }

  private void maybeInitBuilder() {
    if (viaProto || builder == null) {
      builder = RegisterExecutionResultRequestProto.newBuilder(proto);
    }
    viaProto = false;
  }

  @Override
  public int getExitCode() {
    YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProtoOrBuilder p = viaProto ? proto : builder;
    return p.getExitCode();
  }

  @Override
  public void setExitCode(int exitCode) {
    maybeInitBuilder();
    builder.setExitCode(exitCode);
  }


  @Override
  public String getJobName() {
    YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (this.jobName != null) {
      return this.jobName;
    }
    if (!p.hasJobName()) {
      return null;
    }
    this.jobName = p.getJobName();
    return this.jobName;
  }

  @Override
  public void setJobName(String jobName) {
    maybeInitBuilder();
    if (jobName == null) {
      builder.clearJobName();
    }
    this.jobName = jobName;
  }

  @Override
  public String getJobIndex() {
    YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (this.jobIndex != null) {
      return this.jobIndex;
    }
    if (!p.hasJobIndex()) {
      return null;
    }
    this.jobIndex = p.getJobIndex();
    return this.jobIndex;
  }

  @Override
  public void setJobIndex(String jobIndex) {
    maybeInitBuilder();
    if (jobIndex == null) {
      builder.clearJobIndex();
    }
    this.jobIndex = jobIndex;
  }

  @Override
  public String getSessionId() {
    YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (this.sessionId != null) {
      return this.sessionId;
    }
    if (!p.hasSessionId()) {
      return null;
    }
    this.sessionId = p.getSessionId();
    return this.sessionId;
  }

  @Override
  public void setSessionId(String sessionId) {
    maybeInitBuilder();
    if (sessionId == null) {
      builder.clearSessionId();
    }
    this.sessionId = sessionId;
  }
}
