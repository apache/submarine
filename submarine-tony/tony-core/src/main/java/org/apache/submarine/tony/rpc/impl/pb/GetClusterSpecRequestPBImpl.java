/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package org.apache.submarine.tony.rpc.impl.pb;


import org.apache.submarine.tony.rpc.GetClusterSpecRequest;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetClusterSpecRequestProto;

public class GetClusterSpecRequestPBImpl implements GetClusterSpecRequest {
  private GetClusterSpecRequestProto proto = GetClusterSpecRequestProto.getDefaultInstance();
  private GetClusterSpecRequestProto.Builder builder = null;
  private boolean viaProto = false;

  private boolean rebuild = false;

  public GetClusterSpecRequestPBImpl() {
        builder = GetClusterSpecRequestProto.newBuilder();
    }

  public GetClusterSpecRequestPBImpl(GetClusterSpecRequestProto proto) {
    this.proto = proto;
    viaProto = true;
  }

  private void mergeLocalToProto() {
    if (viaProto) {
      maybeInitBuilder();
    }
    proto = builder.build();
    rebuild = false;
    viaProto = true;
  }

  public GetClusterSpecRequestProto getProto() {
     if (rebuild) {
       mergeLocalToProto();
     }
    proto = viaProto ? proto : builder.build();
    viaProto = true;
    return proto;
  }

  private void maybeInitBuilder() {
    if (viaProto || builder == null) {
      builder = GetClusterSpecRequestProto.newBuilder(proto);
    }
    viaProto = false;
  }
}
