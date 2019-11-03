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

package org.apache.submarine.server.submitter.k8s.model;

import com.google.gson.GsonBuilder;
import io.kubernetes.client.models.V1ListMeta;

/**
 * ListMeta describes metadata that synthetic resources must have, including lists and various
 * status objects.
 */
public class ListMeta extends V1ListMeta {
  @Override
  public String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(this);
  }
}
