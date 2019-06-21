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

package org.apache.submarine.tony.util;

import org.apache.submarine.tony.rpc.TaskInfo;
import org.apache.submarine.tony.rpc.impl.TaskStatus;
import org.apache.submarine.tony.rpc.proto.YarnTensorFlowClusterProtos.GetTaskInfosResponseProto.TaskInfoProto;


public class ProtoUtils {
  public static TaskInfo taskInfoProtoToTaskInfo(TaskInfoProto taskInfoProto) {
    TaskInfo taskInfo
        = new TaskInfo(taskInfoProto.getName(), taskInfoProto.getIndex(), taskInfoProto.getUrl());
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
