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

import com.google.gson.JsonSyntaxException;
import io.kubernetes.client.custom.Quantity;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaim;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaimSpec;
import io.kubernetes.client.openapi.models.V1ResourceRequirements;
import io.kubernetes.client.util.generic.options.CreateOptions;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.K8sSubmitter;
import org.apache.submarine.server.submitter.k8s.model.K8sResource;
import org.apache.submarine.server.submitter.k8s.util.NotebookUtils;
import org.apache.submarine.server.submitter.k8s.util.OwnerReferenceUtils;
import org.apache.submarine.server.submitter.k8s.util.YamlUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

public class PersistentVolumeClaim extends V1PersistentVolumeClaim implements
        K8sResource<V1PersistentVolumeClaim> {

  private static final Logger LOG = LoggerFactory.getLogger(PersistentVolumeClaim.class);

  public PersistentVolumeClaim(String namespace, String name, String storage) {
    /*
      Required value
      1. metadata.name
      2. metadata.namespace
      3. spec.accessModes
      4. spec.storageClassName
      5. spec.resources
      Others are not necessary
     */
    V1ObjectMeta pvcMetadata = new V1ObjectMeta();
    pvcMetadata.setNamespace(namespace);
    pvcMetadata.setName(name);
    pvcMetadata.setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
    this.setMetadata(pvcMetadata);

    V1PersistentVolumeClaimSpec pvcSpec = new V1PersistentVolumeClaimSpec();
    pvcSpec.setAccessModes(Collections.singletonList("ReadWriteOnce"));
    pvcSpec.setStorageClassName(NotebookUtils.SC_NAME);
    pvcSpec.setResources(new V1ResourceRequirements().putRequestsItem("storage", new Quantity(storage)));
    this.setSpec(pvcSpec);
  }

  @Override
  public PersistentVolumeClaim read(K8sClient api) {
    return this;
  }

  @Override
  public V1PersistentVolumeClaim create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create PersistentVolumeClaim resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      return api.getPersistentVolumeClaimClient()
          .create(
              this.getMetadata().getNamespace(),
              this, new CreateOptions()
          ).throwsApiException().throwsApiException().getObject();
    } catch (ApiException e) {
      LOG.error("Exception when creating persistent volume claim " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: Create persistent volume claim for " +
              "Notebook object failed by " + e.getMessage());
    }
  }

  @Override
  public V1PersistentVolumeClaim replace(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public V1PersistentVolumeClaim delete(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete PersistentVolumeClaim resource in namespace: {} and name: {}",
                this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      return api.getPersistentVolumeClaimClient()
          .delete(
              this.getMetadata().getNamespace(),
              this.getMetadata().getName()
          ).throwsApiException().getObject();
    } catch (ApiException e) {
      LOG.error("Exception when deleting persistent volume claim " + e.getMessage(), e);
      return (V1PersistentVolumeClaim) K8sSubmitter.API_EXCEPTION_404_CONSUMER.apply(e);
    } catch (JsonSyntaxException e) {
      if (e.getCause() instanceof IllegalStateException) {
        IllegalStateException ise = (IllegalStateException) e.getCause();
        if (ise.getMessage() != null && ise.getMessage().contains("Expected a string but was BEGIN_OBJECT")) {
          LOG.debug("Catching exception because of issue " +
                  "https://github.com/kubernetes-client/java/issues/86", e);
          return this;
        } else {
          throw e;
        }
      } else {
        throw e;
      }
    }
  }
}

