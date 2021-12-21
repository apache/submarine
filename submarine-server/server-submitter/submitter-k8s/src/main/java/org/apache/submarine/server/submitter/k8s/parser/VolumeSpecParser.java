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

package org.apache.submarine.server.submitter.k8s.parser;

import io.kubernetes.client.custom.Quantity;
import io.kubernetes.client.openapi.models.V1HostPathVolumeSource;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PersistentVolume;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaim;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaimSpec;
import io.kubernetes.client.openapi.models.V1PersistentVolumeSpec;
import io.kubernetes.client.openapi.models.V1ResourceRequirements;

import java.util.Collections;

public class VolumeSpecParser {
  public static V1PersistentVolume parsePersistentVolume(String name, String hostPath, String storage) {
    V1PersistentVolume pv = new V1PersistentVolume();
    /*
      Required value
      1. metadata.name
      2. spec.accessModes
      3. spec.capacity
      4. spec.storageClassName
      Others are not necessary
     */

    V1ObjectMeta pvMetadata = new V1ObjectMeta();
    pvMetadata.setName(name);
    pv.setMetadata(pvMetadata);

    V1PersistentVolumeSpec pvSpec = new V1PersistentVolumeSpec();
    pvSpec.setAccessModes(Collections.singletonList("ReadWriteMany"));
    pvSpec.setCapacity(Collections.singletonMap("storage", new Quantity(storage)));
    pvSpec.setStorageClassName("standard");
    pvSpec.setHostPath(new V1HostPathVolumeSource().path(hostPath));
    pv.setSpec(pvSpec);

    return pv;
  }

  public static V1PersistentVolumeClaim parsePersistentVolumeClaim(
      String name, String scName, String storage) {
    V1PersistentVolumeClaim pvc = new V1PersistentVolumeClaim();
    /*
      Required value
      1. metadata.name
      2. spec.accessModes
      3. spec.storageClassName
      4. spec.resources
      Others are not necessary
     */

    V1ObjectMeta pvcMetadata = new V1ObjectMeta();
    pvcMetadata.setName(name);
    pvc.setMetadata(pvcMetadata);

    V1PersistentVolumeClaimSpec pvcSpec = new V1PersistentVolumeClaimSpec();
    pvcSpec.setAccessModes(Collections.singletonList("ReadWriteOnce"));
    pvcSpec.setStorageClassName(scName);
    pvcSpec.setResources(new V1ResourceRequirements().putRequestsItem("storage", new Quantity(storage)));
    pvc.setSpec(pvcSpec);

    return pvc;
  }
}
