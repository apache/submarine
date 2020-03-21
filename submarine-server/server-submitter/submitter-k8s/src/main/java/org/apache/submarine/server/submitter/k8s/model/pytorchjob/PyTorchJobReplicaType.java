/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.submarine.server.submitter.k8s.model.pytorchjob;

import com.google.gson.annotations.SerializedName;
import org.apache.submarine.server.submitter.k8s.model.MLJobReplicaType;

public enum PyTorchJobReplicaType implements MLJobReplicaType {

  @SerializedName("Master")
  Master("Master"),

  @SerializedName("Worker")
  Worker("Worker");

  private String typeName;

  PyTorchJobReplicaType(String n) {
    this.typeName = n;
  }

  public static boolean isSupportedReplicaType(String type) {
    return type.equalsIgnoreCase("Master") ||
        type.equalsIgnoreCase("Worker");
  }

  public static String[] names() {
    PyTorchJobReplicaType[] types = values();
    String[] names = new String[types.length];
    for (int i = 0; i < types.length; i++) {
      names[i] = types[i].name();
    }
    return names;
  }

  @Override
  public String getTypeName() {
    return this.typeName;
  }
}
