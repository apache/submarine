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

import org.apache.submarine.tony.rpc.RegisterTensorBoardUrlResponse;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterTensorBoardUrlResponseProto;


public class RegisterTensorBoardUrlResponsePBImpl implements RegisterTensorBoardUrlResponse {
  private RegisterTensorBoardUrlResponseProto proto
      = RegisterTensorBoardUrlResponseProto.getDefaultInstance();
  private RegisterTensorBoardUrlResponseProto.Builder builder = null;
  private boolean viaProto = false;

  private String spec = null;

  public RegisterTensorBoardUrlResponsePBImpl() {
    builder = RegisterTensorBoardUrlResponseProto.newBuilder();
  }

  public RegisterTensorBoardUrlResponsePBImpl(RegisterTensorBoardUrlResponseProto proto) {
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
    if (this.spec != null) {
      builder.setSpec(this.spec);
    }
  }

  public RegisterTensorBoardUrlResponseProto getProto() {
    mergeLocalToProto();
    proto = viaProto ? proto : builder.build();
    viaProto = true;
    return proto;
  }

  private void maybeInitBuilder() {
    if (viaProto || builder == null) {
      builder = RegisterTensorBoardUrlResponseProto.newBuilder(proto);
    }
    viaProto = false;
  }
  @Override
  public String getSpec() {
    YarnTensorFlowClusterProtos.RegisterTensorBoardUrlResponseProtoOrBuilder p = viaProto ? proto : builder;
    if (this.spec != null) {
      return this.spec;
    }
    if (!p.hasSpec()) {
      return null;
    }
    this.spec = p.getSpec();
    return this.spec;
  }

  @Override
  public void setSpec(String spec) {
    maybeInitBuilder();
    if (spec == null) {
      builder.clearSpec();
    }
    this.spec = spec;
  }
}
