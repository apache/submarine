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

import org.apache.submarine.tony.rpc.RegisterExecutionResultResponse;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterExecutionResultResponseProto;


public class RegisterExecutionResultResponsePBImpl implements RegisterExecutionResultResponse {
  private RegisterExecutionResultResponseProto proto
      = RegisterExecutionResultResponseProto.getDefaultInstance();
  private RegisterExecutionResultResponseProto.Builder builder = null;
  private boolean viaProto = false;

  private String message = null;

  public RegisterExecutionResultResponsePBImpl() {
    builder = RegisterExecutionResultResponseProto.newBuilder();
  }

  public RegisterExecutionResultResponsePBImpl(RegisterExecutionResultResponseProto proto) {
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
    if (this.message != null) {
      builder.setMessage(this.message);
    }
  }

  public RegisterExecutionResultResponseProto getProto() {
    mergeLocalToProto();
    proto = viaProto ? proto : builder.build();
    viaProto = true;
    return proto;
  }

  private void maybeInitBuilder() {
    if (viaProto || builder == null) {
      builder = RegisterExecutionResultResponseProto.newBuilder(proto);
    }
    viaProto = false;
  }
  @Override
  public String getMessage() {
    YarnTensorFlowClusterProtos.RegisterExecutionResultResponseProtoOrBuilder p = viaProto ? proto : builder;
    if (this.message != null) {
      return this.message;
    }
    if (!p.hasMessage()) {
      return null;
    }
    this.message = p.getMessage();
    return this.message;
  }

  @Override
  public void setMessage(String message) {
    maybeInitBuilder();
    if (message == null) {
      builder.clearMessage();
    }
    this.message = message;
  }
}
