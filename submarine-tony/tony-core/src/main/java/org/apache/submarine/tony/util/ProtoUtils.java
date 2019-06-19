/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package org.apache.submarine.tony.util;

import org.apache.submarine.tony.rpc.TaskInfo;
import org.apache.submarine.tony.rpc.impl.TaskStatus;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetTaskInfosResponseProto.TaskInfoProto;


public class ProtoUtils {
  public static TaskInfo taskInfoProtoToTaskInfo(TaskInfoProto taskInfoProto) {
    TaskInfo taskInfo = new TaskInfo(taskInfoProto.getName(), taskInfoProto.getIndex(), taskInfoProto.getUrl());
    taskInfo.setState(TaskStatus.values()[taskInfoProto.getTaskStatus().ordinal()]);
    return taskInfo;
  }

  public static TaskInfoProto taskInfoToTaskInfoProto(TaskInfo taskInfo) {
    return TaskInfoProto.newBuilder()
        .setName(taskInfo.getName())
        .setIndex(taskInfo.getIndex())
        .setUrl(taskInfo.getUrl())
        .setTaskStatus(TaskInfoProto.TaskStatus.values()[taskInfo.getStatus().ordinal()]).build();
  }

  private ProtoUtils() { }
}
