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

package org.apache.submarine.server.submitter.k8s.model.seldon;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.util.generic.options.CreateOptions;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.serve.tensorflow.SeldonTFServing;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.model.istio.IstioVirtualService;
import org.apache.submarine.server.submitter.k8s.util.YamlUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;

import static org.apache.submarine.server.submitter.k8s.K8sSubmitter.getDeleteOptions;

/**
 * Seldon Deployment Tensorflow Serving Resource
 */
public class SeldonDeploymentTFServing extends SeldonTFServing implements SeldonResource {

  private static final Logger LOG = LoggerFactory.getLogger(SeldonDeploymentTFServing.class);

  private String modelName;
  private Integer modelVersion;

  public SeldonDeploymentTFServing() {
  }

  public SeldonDeploymentTFServing(String resourceName, String modelName, Integer modelVersion,
                                   String modelURI) {
    super(resourceName, modelName, modelURI);
    this.modelName = modelName;
    this.modelVersion = modelVersion;
  }

  @Override
  public SeldonTFServing read(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public SeldonTFServing create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create Seldon TFServing resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      api.getSeldonDeploymentClient()
          .create(getMetadata().getNamespace(), this, new CreateOptions())
          .throwsApiException();
      return this;
    } catch (ApiException e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public SeldonTFServing replace(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public SeldonTFServing delete(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete Seldon TFServing resource in namespace: {} and name: {}",
            this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      api.getSeldonDeploymentClient()
          .delete(getMetadata().getNamespace(), getMetadata().getName(),
              getDeleteOptions(getApiVersion()))
          .throwsApiException();
      return this;
    } catch (ApiException e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public IstioVirtualService getIstioVirtualService() {
    IstioVirtualService service = new IstioVirtualService(getMetadata().getName(), this.modelVersion);
    service.getMetadata()
        .setLabels(new HashMap<String, String>() {{ put(MODEL_NAME_LABEL, modelName); }});
    return service;
  }
}
