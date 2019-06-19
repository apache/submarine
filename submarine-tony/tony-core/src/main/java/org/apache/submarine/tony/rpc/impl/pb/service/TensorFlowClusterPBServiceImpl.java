/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package org.apache.submarine.tony.rpc.impl.pb.service;

import com.google.protobuf.RpcController;
import com.google.protobuf.ServiceException;
import org.apache.submarine.tony.rpc.Empty;
import org.apache.submarine.tony.rpc.GetClusterSpecResponse;
import org.apache.submarine.tony.rpc.GetTaskInfosResponse;
import org.apache.submarine.tony.rpc.HeartbeatResponse;
import org.apache.submarine.tony.rpc.RegisterExecutionResultResponse;
import org.apache.submarine.tony.rpc.RegisterTensorBoardUrlResponse;
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
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.EmptyProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetClusterSpecRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetClusterSpecResponseProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetTaskInfosRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterExecutionResultRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterExecutionResultResponseProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterTensorBoardUrlRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterTensorBoardUrlResponseProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterWorkerSpecRequestProto;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.RegisterWorkerSpecResponseProto;
import org.apache.hadoop.yarn.exceptions.YarnException;

import java.io.IOException;

public class TensorFlowClusterPBServiceImpl implements TensorFlowClusterPB {
  private TensorFlowCluster real;

  public TensorFlowClusterPBServiceImpl(TensorFlowCluster impl) {
    this.real = impl;
  }

  @Override
  public YarnTensorFlowClusterProtos.GetTaskInfosResponseProto getTaskInfos(RpcController controller,
                                                                           GetTaskInfosRequestProto proto) throws ServiceException {
    GetTaskInfosRequestPBImpl request = new GetTaskInfosRequestPBImpl(proto);
    try {
      GetTaskInfosResponse response = real.getTaskInfos(request);
      return ((GetTaskInfosResponsePBImpl) response).getProto();
    } catch (YarnException | IOException e) {
      throw new ServiceException(e);
    }
  }

  @Override
  public GetClusterSpecResponseProto getClusterSpec(RpcController controller,
                                                    GetClusterSpecRequestProto proto) throws ServiceException {
    GetClusterSpecRequestPBImpl request = new GetClusterSpecRequestPBImpl(proto);
    try {
      GetClusterSpecResponse response = real.getClusterSpec(request);
      return ((GetClusterSpecResponsePBImpl) response).getProto();
    } catch (YarnException | IOException e) {
      throw new ServiceException(e);
    }
  }

  @Override
  public RegisterWorkerSpecResponseProto registerWorkerSpec(RpcController controller,
                                                            RegisterWorkerSpecRequestProto proto) throws ServiceException {
    RegisterWorkerSpecRequestPBImpl request = new RegisterWorkerSpecRequestPBImpl(proto);
    try {
      RegisterWorkerSpecResponse response = real.registerWorkerSpec(request);
      return ((RegisterWorkerSpecResponsePBImpl) response).getProto();
    } catch (YarnException | IOException e) {
      throw new ServiceException(e);
    }
  }

  @Override
  public RegisterTensorBoardUrlResponseProto registerTensorBoardUrl(
      RpcController controller, RegisterTensorBoardUrlRequestProto proto)
      throws ServiceException {
    RegisterTensorBoardUrlRequestPBImpl request = new RegisterTensorBoardUrlRequestPBImpl(proto);
    try {
      RegisterTensorBoardUrlResponse response = real.registerTensorBoardUrl(request);
      return ((RegisterTensorBoardUrlResponsePBImpl) response).getProto();
    } catch (Exception e) {
      throw new ServiceException(e);
    }
  }

  @Override
  public RegisterExecutionResultResponseProto registerExecutionResult(
      RpcController controller, RegisterExecutionResultRequestProto proto)
      throws ServiceException {
    RegisterExecutionResultRequestPBImpl request = new RegisterExecutionResultRequestPBImpl(proto);
    try {
      RegisterExecutionResultResponse response = real.registerExecutionResult(request);
      return ((RegisterExecutionResultResponsePBImpl) response).getProto();
    } catch (Exception e) {
      throw new ServiceException(e);
    }
  }

  @Override
  public EmptyProto finishApplication(RpcController controller, EmptyProto proto)
      throws ServiceException {
    EmptyPBImpl request = new EmptyPBImpl(proto);
    try {
      Empty response = real.finishApplication(request);
      return ((EmptyPBImpl) response).getProto();
    } catch (Exception e) {
      throw new ServiceException(e);
    }
  }

  @Override
  public YarnTensorFlowClusterProtos.HeartbeatResponseProto taskExecutorHeartbeat(RpcController controller,
      YarnTensorFlowClusterProtos.HeartbeatRequestProto proto) throws ServiceException {
    HeartbeatRequestPBImpl request = new HeartbeatRequestPBImpl(proto);
    try {
      HeartbeatResponse response = real.taskExecutorHeartbeat(request);
      return ((HeartbeatResponsePBImpl) response).getProto();
    } catch (Exception e) {
      throw new ServiceException(e);
    }
  }
}
