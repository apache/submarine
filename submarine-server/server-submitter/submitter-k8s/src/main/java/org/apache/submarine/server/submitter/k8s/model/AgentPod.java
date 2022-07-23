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

import java.util.ArrayList;
import java.util.List;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.common.CustomResourceType;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1ObjectMetaBuilder;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodSpec;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.util.YamlUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AgentPod extends V1Pod implements K8sResource<AgentPod> {

  private static final Logger LOG = LoggerFactory.getLogger(AgentPod.class);

  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
  private static final String AGENT_IMAGE = "apache/submarine:agent-0.8.0-SNAPSHOT";
  private static final String CONTAINER_NAME = "agent";

  public AgentPod(String namespace, String name,
                  CustomResourceType type,
                  String resourceId) {
    super();

    V1ObjectMetaBuilder metaBuilder = new V1ObjectMetaBuilder();
    metaBuilder.withName(getNormalizePodName(type, name, resourceId))
        .withNamespace(namespace)
        .addToLabels("app", type.toString().toLowerCase())
        // There is no need to add istio sidecar. Otherwise, the pod may not end normally
        // https://istio.io/latest/docs/setup/additional-setup/sidecar-injection/
        // Controlling the injection policy Section
        .addToAnnotations("sidecar.istio.io/inject", "false");
    this.setMetadata(metaBuilder.build());

    V1PodSpec spec = new V1PodSpec();
    List<V1Container> containers = spec.getContainers();
    V1Container agentContainer = new V1Container();
    agentContainer.setName(CONTAINER_NAME);
    agentContainer.setImage(AGENT_IMAGE);

    List<V1EnvVar> envVarList = new ArrayList<>();
    V1EnvVar crTypeVar = new V1EnvVar();
    crTypeVar.setName("CUSTOM_RESOURCE_TYPE");
    crTypeVar.setValue(type.toString());

    V1EnvVar crNameVar = new V1EnvVar();
    crNameVar.setName("CUSTOM_RESOURCE_NAME");
    crNameVar.setValue(name);

    V1EnvVar namespaceVar = new V1EnvVar();
    namespaceVar.setName("NAMESPACE");
    namespaceVar.setValue(namespace);

    V1EnvVar serverHostVar = new V1EnvVar();
    serverHostVar.setName("SERVER_HOST");
    serverHostVar.setValue(conf.getServerServiceName());

    V1EnvVar serverPortVar = new V1EnvVar();
    serverPortVar.setName("SERVER_PORT");
    serverPortVar.setValue(String.valueOf(conf.getServerPort()));

    V1EnvVar customResourceIdVar = new V1EnvVar();
    customResourceIdVar.setName("CUSTOM_RESOURCE_ID");
    customResourceIdVar.setValue(resourceId);

    envVarList.add(crTypeVar);
    envVarList.add(crNameVar);
    envVarList.add(namespaceVar);
    envVarList.add(serverHostVar);
    envVarList.add(serverPortVar);
    envVarList.add(customResourceIdVar);

    agentContainer.env(envVarList);

    containers.add(agentContainer);

    spec.setRestartPolicy("OnFailure");
    this.setSpec(spec);
  }

  private String getNormalizePodName(CustomResourceType type, String name, String resourceId) {
    return String.format("%s-%s-%s-%s", resourceId.toString().toLowerCase().replace('_', '-'),
            type.toString().toLowerCase(), name, CONTAINER_NAME);
  }

  @Override
  public AgentPod read(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public AgentPod create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create AgentPod resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      api.getPodClient().create(this).throwsApiException();
    } catch (ApiException e) {
      LOG.error("K8s submitter: create AgentPod object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: create AgentPod object failed by " +
              e.getMessage());
    }
    return this;
  }

  @Override
  public AgentPod replace(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public AgentPod delete(K8sClient api) {
    if (LOG.isDebugEnabled()) {
      LOG.debug("Delete AgentPod resource in namespace: {} and name: {}",
              this.getMetadata().getNamespace(), this.getMetadata().getName());
    }
    api.getPodClient().delete(this.getMetadata().getNamespace(), this.getMetadata().getName());
    return this;
  }
}
