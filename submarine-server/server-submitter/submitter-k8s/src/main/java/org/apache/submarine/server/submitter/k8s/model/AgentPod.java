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
import org.apache.submarine.server.api.common.CustomResourceType;

import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodSpec;

public class AgentPod extends V1Pod{
  private static SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
  private static final String AGENT_IMAGE = "apache/submarine:agent-0.7.0";
  private static final String CONTAINER_NAME = "agent";
  public AgentPod(String namespace, String name,
          CustomResourceType type,
          String resourceId) {
    super();
    V1ObjectMeta meta = new V1ObjectMeta();

    meta.setName(
            String.format("%s-%s-%s-%s", type.toString().toLowerCase(), name,
                    resourceId.toLowerCase(), CONTAINER_NAME));
    meta.setNamespace(namespace);
    this.setMetadata(meta);

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
}
