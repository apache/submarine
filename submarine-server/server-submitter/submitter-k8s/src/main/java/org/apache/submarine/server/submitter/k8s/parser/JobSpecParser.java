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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.kubernetes.client.custom.Quantity;
import io.kubernetes.client.models.V1Container;
import io.kubernetes.client.models.V1EnvVar;
import io.kubernetes.client.models.V1ObjectMeta;
import io.kubernetes.client.models.V1PodSpec;
import io.kubernetes.client.models.V1PodTemplateSpec;
import io.kubernetes.client.models.V1ResourceRequirements;
import org.apache.submarine.server.api.spec.JobLibrarySpec;
import org.apache.submarine.server.api.spec.JobSpec;
import org.apache.submarine.server.api.spec.JobTaskSpec;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobSpec;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFReplicaSpec;

public class JobSpecParser {
  /**
   * Parse the job spec to {@link TFJob}
   * @param jobSpec job spec
   * @return the TFJob object
   */
  public static TFJob parseTFJob(JobSpec jobSpec) {
    TFJob tfJob = new TFJob();
    tfJob.setApiVersion(jobSpec.getSubmitterSpec().getApiVersion());
    tfJob.setMetadata(parseTFMetadata(jobSpec));
    tfJob.setSpec(parseTFJobSpec(jobSpec));
    return tfJob;
  }

  private static V1ObjectMeta parseTFMetadata(JobSpec jobSpec) {
    V1ObjectMeta meta = new V1ObjectMeta();
    meta.setNamespace(jobSpec.getSubmitterSpec().getNamespace());
    meta.setName(jobSpec.getName());
    return meta;
  }

  private static TFJobSpec parseTFJobSpec(JobSpec jobSpec) {
    TFJobSpec tfJobSpec = new TFJobSpec();
    Map<String, TFReplicaSpec> replicaSpecMap = new HashMap<>();
    for (Map.Entry<String, JobTaskSpec> entry : jobSpec.getTaskSpecs().entrySet()) {
      TFReplicaSpec spec = new TFReplicaSpec();
      spec.setReplicas(entry.getValue().getReplicas());
      spec.setTemplate(parseTemplateSpec(entry.getValue(), jobSpec.getLibrarySpec()));
      replicaSpecMap.put(entry.getValue().getName(), spec);
    }
    tfJobSpec.setTfReplicaSpecs(replicaSpecMap);
    return tfJobSpec;
  }

  private static V1PodTemplateSpec parseTemplateSpec(JobTaskSpec taskSpec, JobLibrarySpec libSpec) {
    V1PodTemplateSpec templateSpec = new V1PodTemplateSpec();
    V1PodSpec podSpec = new V1PodSpec();
    List<V1Container> containers = new ArrayList<>();
    V1Container container = new V1Container();
    container.setName(libSpec.getName().toLowerCase());
    // image
    if (taskSpec.getImage() != null) {
      container.setImage(taskSpec.getImage());
    } else {
      container.setImage(libSpec.getImage());
    }
    // cmd
    if (taskSpec.getCmd() != null) {
      container.setCommand(Arrays.asList(taskSpec.getCmd().split(" ")));
    } else {
      container.setCommand(Arrays.asList(libSpec.getCmd().split(" ")));
    }
    // resources
    V1ResourceRequirements resources = new V1ResourceRequirements();
    resources.setLimits(parseResources(taskSpec));
    container.setResources(resources);
    container.setEnv(parseEnvVars(taskSpec, libSpec.getEnvVars()));
    podSpec.setContainers(containers);
    templateSpec.setSpec(podSpec);
    return templateSpec;
  }

  private static List<V1EnvVar> parseEnvVars(JobTaskSpec spec, Map<String, String> defaultEnvs) {
    if (spec.getEnvVars() != null) {
      return parseEnvVars(spec.getEnvVars());
    }
    return parseEnvVars(defaultEnvs);
  }

  private static List<V1EnvVar> parseEnvVars(Map<String, String> envMap) {
    List<V1EnvVar> envVars = new ArrayList<>();
    for (Map.Entry<String, String> entry : envMap.entrySet()) {
      V1EnvVar env = new V1EnvVar();
      env.setName(entry.getKey());
      env.setValue(entry.getValue());
      envVars.add(env);
    }
    return envVars;
  }

  private static Map<String, Quantity> parseResources(JobTaskSpec taskSpec) {
    Map<String, Quantity> resources = new HashMap<>();
    if (taskSpec.getCpu() != null) {
      resources.put("cpu", new Quantity(taskSpec.getCpu()));
    }
    if (taskSpec.getMemory() != null) {
      resources.put("memory", new Quantity(taskSpec.getMemory()));
    }
    if (taskSpec.getGpu() != null) {
      resources.put("nvidia.com/gpu", new Quantity(taskSpec.getGpu()));
    }
    return resources;
  }
}
