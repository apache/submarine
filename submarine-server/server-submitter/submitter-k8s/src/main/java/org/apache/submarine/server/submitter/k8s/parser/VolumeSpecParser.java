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
import io.kubernetes.client.models.V1HostPathVolumeSource;
import io.kubernetes.client.models.V1ObjectMeta;
import io.kubernetes.client.models.V1PersistentVolume;
import io.kubernetes.client.models.V1PersistentVolumeClaim;
import io.kubernetes.client.models.V1PersistentVolumeClaimSpec;
import io.kubernetes.client.models.V1PersistentVolumeSpec;
import io.kubernetes.client.models.V1ResourceRequirements;

import java.util.Collections;

public class VolumeSpecParser {
  public static V1PersistentVolume parsePersistentVolume(String name, String host_path, String storage) {
    V1PersistentVolume pv = new V1PersistentVolume();
    /*
      Required value
      1. metadata.name
      2. spec.accessModes
      3. spec.capacity
      4. spec.storageClassName
      Others are not necessary
     */

    V1ObjectMeta pv_metadata = new V1ObjectMeta();
    pv_metadata.setName(name);
    pv.setMetadata(pv_metadata);

    V1PersistentVolumeSpec pv_spec = new V1PersistentVolumeSpec();
    pv_spec.setAccessModes(Collections.singletonList("ReadWriteMany"));
    pv_spec.setCapacity(Collections.singletonMap("storage", new Quantity(storage)));
    pv_spec.setStorageClassName("standard");
    pv_spec.setHostPath(new V1HostPathVolumeSource().path(host_path));
    pv.setSpec(pv_spec);

    return pv;
  }

  public static V1PersistentVolumeClaim parsePersistentVolumeClaim(
      String name, String volume, String storage) {
    V1PersistentVolumeClaim pvc = new V1PersistentVolumeClaim();
    /*
      Required value
      1. metadata.name
      2. spec.accessModes
      3. spec.storageClassName
      4. spec.resources
      Others are not necessary
     */

    V1ObjectMeta pvc_metadata = new V1ObjectMeta();
    pvc_metadata.setName(name);
    pvc.setMetadata(pvc_metadata);

    V1PersistentVolumeClaimSpec pvc_spec = new V1PersistentVolumeClaimSpec();
    pvc_spec.setAccessModes(Collections.singletonList("ReadWriteMany"));
    pvc_spec.setStorageClassName("standard");
    pvc_spec.setResources(new V1ResourceRequirements().putRequestsItem("storage", new Quantity(storage)));
    pvc_spec.setVolumeName(volume); // bind pvc to specific pv
    pvc.setSpec(pvc_spec);

    return pvc;
  }

}
