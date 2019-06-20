/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */

package org.apache.submarine.tony.rpc.impl.pb.client;

import com.google.protobuf.ServiceException;
import org.apache.submarine.tony.rpc.Empty;
import org.apache.submarine.tony.rpc.GetClusterSpecRequest;
import org.apache.submarine.tony.rpc.GetClusterSpecResponse;
import org.apache.submarine.tony.rpc.GetTaskInfosRequest;
import org.apache.submarine.tony.rpc.GetTaskInfosResponse;
import org.apache.submarine.tony.rpc.HeartbeatRequest;
import org.apache.submarine.tony.rpc.HeartbeatResponse;
import org.apache.submarine.tony.rpc.RegisterExecutionResultRequest;
import org.apache.submarine.tony.rpc.RegisterExecutionResultResponse;
import org.apache.submarine.tony.rpc.RegisterTensorBoardUrlRequest;
import org.apache.submarine.tony.rpc.RegisterTensorBoardUrlResponse;
import org.apache.submarine.tony.rpc.RegisterWorkerSpecRequest;
import org.apache.submarine.tony.rpc.RegisterWorkerSpecResponse;
import org.apache.submarine.tony.rpc.TensorFlowCluster;
import org.apache.submarine.tony.rpc.TensorFlowClusterPB;
import org.apache.submarine.tony.rpc.impl.pb.EmptyPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.GetClusterSpecRequestPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.GetClusterSpecResponsePBImpl;
import org.apache.submarine.tony.rpc.impl.pb.GetTaskInfosRequestPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.GetTaskInfosResponsePBImpl;
import org.apache.submarine.tony.rpc.impl.pb.HeartbeatRequestPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.HeartbeatResponsePBImpl;
import org.apache.submarine.tony.rpc.impl.pb.RegisterExecutionResultRequestPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.RegisterExecutionResultResponsePBImpl;
import org.apache.submarine.tony.rpc.impl.pb.RegisterTensorBoardUrlRequestPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.RegisterTensorBoardUrlResponsePBImpl;
import org.apache.submarine.tony.rpc.impl.pb.RegisterWorkerSpecRequestPBImpl;
import org.apache.submarine.tony.rpc.impl.pb.RegisterWorkerSpecResponsePBImpl;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetClusterSpecRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetTaskInfosRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterWorkerSpecRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.HeartbeatRequestProto;
import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.ipc.ProtobufRpcEngine;
import org.apache.hadoop.ipc.ProtocolSignature;
import org.apache.hadoop.ipc.RPC;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.ipc.RPCUtil;


/**
 * This class is instantiated using reflection by YARN's RecordFactoryPBImpl
 */
public class TensorFlowClusterPBClientImpl implements TensorFlowCluster, Closeable {
  private TensorFlowClusterPB proxy;

  public TensorFlowClusterPBClientImpl(long clientVersion,
                                       InetSocketAddress addr,
                                       Configuration conf)
      throws IOException {
    RPC.setProtocolEngine(conf, TensorFlowClusterPB.class, ProtobufRpcEngine.class);
    proxy = RPC.getProxy(TensorFlowClusterPB.class, clientVersion, addr, conf);
  }

  @Override
  public void close() {
    if (this.proxy != null) {
      RPC.stopProxy(this.proxy);
    }
  }

  @Override
  public GetTaskInfosResponse getTaskInfos(GetTaskInfosRequest request)
      throws IOException, YarnException {
    GetTaskInfosRequestProto requestProto = ((GetTaskInfosRequestPBImpl) request).getProto();
    try {
      return new GetTaskInfosResponsePBImpl(proxy.getTaskInfos(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public GetClusterSpecResponse getClusterSpec(GetClusterSpecRequest request)
      throws YarnException, IOException {
    GetClusterSpecRequestProto requestProto = ((GetClusterSpecRequestPBImpl) request).getProto();
    try {
      return new GetClusterSpecResponsePBImpl(proxy.getClusterSpec(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public RegisterWorkerSpecResponse registerWorkerSpec(RegisterWorkerSpecRequest request)
      throws YarnException, IOException {
    RegisterWorkerSpecRequestProto requestProto = ((RegisterWorkerSpecRequestPBImpl) request).getProto();
    try {
      return new RegisterWorkerSpecResponsePBImpl(proxy.registerWorkerSpec(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public RegisterTensorBoardUrlResponse registerTensorBoardUrl(RegisterTensorBoardUrlRequest request)
      throws YarnException, IOException {
    YarnTensorFlowClusterProtos.RegisterTensorBoardUrlRequestProto requestProto
        = ((RegisterTensorBoardUrlRequestPBImpl) request).getProto();
    try {
      return new RegisterTensorBoardUrlResponsePBImpl(proxy.registerTensorBoardUrl(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public RegisterExecutionResultResponse registerExecutionResult(RegisterExecutionResultRequest request)
      throws YarnException, IOException {
    YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProto requestProto
        = ((RegisterExecutionResultRequestPBImpl) request).getProto();
    try {
      return new RegisterExecutionResultResponsePBImpl(proxy.registerExecutionResult(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public Empty finishApplication(Empty request) throws YarnException, IOException {
    YarnTensorFlowClusterProtos.EmptyProto requestProto = ((EmptyPBImpl) request).getProto();
    try {
      return new EmptyPBImpl(proxy.finishApplication(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public HeartbeatResponse taskExecutorHeartbeat(HeartbeatRequest request) throws YarnException, IOException {
    HeartbeatRequestProto requestProto = ((HeartbeatRequestPBImpl) request).getProto();
    try {
      return new HeartbeatResponsePBImpl(proxy.taskExecutorHeartbeat(null, requestProto));
    } catch (ServiceException e) {
      RPCUtil.unwrapAndThrowException(e);
      return null;
    }
  }

  @Override
  public long getProtocolVersion(String protocol, long version) {
    return TensorFlowCluster.versionID;
  }

  @Override
  public ProtocolSignature getProtocolSignature(String protocol,
                                                long clientVersion,
                                                int clientMethodsHash) throws IOException {
    return ProtocolSignature.getProtocolSignature(this,
        protocol, clientVersion, clientMethodsHash);
  }
}
