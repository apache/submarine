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

package org.apache.submarine.server.submitter.k8s.model.istio;

import com.google.gson.JsonSyntaxException;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.util.generic.options.CreateOptions;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.serve.istio.IstioVirtualServiceSpec;
import org.apache.submarine.serve.utils.IstioConstants;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.K8sSubmitter;
import org.apache.submarine.server.submitter.k8s.model.K8sResource;
import org.apache.submarine.server.submitter.k8s.util.YamlUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.submarine.server.submitter.k8s.K8sSubmitter.getDeleteOptions;

public class IstioVirtualService extends org.apache.submarine.serve.istio.IstioVirtualService
        implements K8sResource<IstioVirtualService> {

  private static final Logger LOG = LoggerFactory.getLogger(IstioVirtualService.class);

  public IstioVirtualService(V1ObjectMeta metadata, IstioVirtualServiceSpec spec) {
    super(metadata, spec);
  }

  public IstioVirtualService(String modelName, Integer modelVersion) {
    super(modelName, modelVersion);
  }

  public IstioVirtualService(V1ObjectMeta metadata) {
    super(metadata);
  }

  @Override
  public IstioVirtualService read(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public IstioVirtualService create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create VirtualService resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      api.getIstioVirtualServiceClient()
        .create(
            getMetadata().getNamespace(),
            this,
            new CreateOptions()
        ).throwsApiException();
      return this;
    } catch (ApiException e) {
      LOG.error("K8s submitter: Create notebook VirtualService custom resource object failed by " +
              e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    }
  }

  @Override
  public IstioVirtualService replace(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public IstioVirtualService delete(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete VirtualService resource in namespace: {} and name: {}",
                this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      api.getIstioVirtualServiceClient()
        .delete(
          this.getMetadata().getNamespace(),
          this.getMetadata().getName(),
          getDeleteOptions(IstioConstants.API_VERSION)
        ).throwsApiException();
    } catch (ApiException e) {
      LOG.error("K8s submitter: Delete notebook VirtualService custom resource object failed by " +
              e.getMessage(), e);
      K8sSubmitter.API_EXCEPTION_404_CONSUMER.apply(e);
    }
    return this;
  }
}
