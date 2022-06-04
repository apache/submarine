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

package org.apache.submarine.server.submitter.k8s.model.common;

import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.model.K8sResource;

public class NullResource implements K8sResource<NullResource> {

  @Override
  public String getKind() {
    throw new UnsupportedOperationException();
  }

  @Override
  public V1ObjectMeta getMetadata() {
    throw new UnsupportedOperationException();
  }

  @Override
  public NullResource read(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public NullResource create(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public NullResource replace(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public NullResource delete(K8sClient api) {
    throw new UnsupportedOperationException();
  }
}
