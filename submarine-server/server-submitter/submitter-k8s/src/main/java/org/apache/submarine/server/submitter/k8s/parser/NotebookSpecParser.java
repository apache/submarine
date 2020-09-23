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
import io.kubernetes.client.models.V1Container;
import io.kubernetes.client.models.V1EnvVar;
import io.kubernetes.client.models.V1ObjectMeta;
import io.kubernetes.client.models.V1PodTemplateSpec;
import io.kubernetes.client.models.V1PodSpec;
import io.kubernetes.client.models.V1ResourceRequirements;

import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.api.spec.NotebookPodSpec;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.environment.EnvironmentManager;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.NotebookCRSpec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NotebookSpecParser {

  private static SubmarineConfiguration conf =
          SubmarineConfiguration.getInstance();


  public static NotebookCR parseNotebook(NotebookSpec spec) {
    NotebookCR notebookCR = new NotebookCR();
    notebookCR.setMetadata(parseMetadata(spec));
    notebookCR.setSpec(parseNotebookCRSpec(spec));
    return notebookCR;
  }

  private static V1ObjectMeta parseMetadata(NotebookSpec spec) {
    V1ObjectMeta meta = new V1ObjectMeta();
    meta.setName(spec.getMeta().getName());
    meta.setNamespace(spec.getMeta().getNamespace());
    return meta;
  }

  private static NotebookCRSpec parseNotebookCRSpec(NotebookSpec spec) {
    NotebookCRSpec CRSpec = new NotebookCRSpec();
    CRSpec.setTemplate(parseTemplateSpec(spec));
    return CRSpec;
  }

  private static V1PodTemplateSpec parseTemplateSpec(NotebookSpec notebookSpec) {
    NotebookPodSpec notebookPodSpec = notebookSpec.getSpec();
    V1PodTemplateSpec podTemplateSpec = new V1PodTemplateSpec();
    V1PodSpec podSpec = new V1PodSpec();
    // Set container
    List<V1Container> containers = new ArrayList<>();
    V1Container container = new V1Container();
    container.setName(notebookSpec.getMeta().getName());

    // Environment variables
    if (notebookPodSpec.getEnvVars() != null) {
      container.setEnv(parseEnvVars(notebookPodSpec));
    }

    // Environment
    if (getEnvironment(notebookSpec) != null) {
      EnvironmentSpec environmentSpec = getEnvironment(notebookSpec).getEnvironmentSpec();
      String baseImage = environmentSpec.getDockerImage();
      KernelSpec kernel = environmentSpec.getKernelSpec();
      container.setImage(baseImage);

      String condaVersionValidationCommand = generateCondaVersionValidateCommand();
      StringBuffer installCommand = new StringBuffer();
      installCommand.append(condaVersionValidationCommand);

      // If dependencies isn't empty
      if (kernel.getDependencies().size() > 0) {
        installCommand.append(" && conda install -y");
        for (String channel : kernel.getChannels()) {
          installCommand.append(" ");
          installCommand.append("-c");
          installCommand.append(" ");
          installCommand.append(channel);
        }
        for (String dependency : kernel.getDependencies()) {
          installCommand.append(" ");
          installCommand.append(dependency);
        }
      }
      V1EnvVar installCommandEnv = new V1EnvVar();
      installCommandEnv.setName("INSTALL_ENVIRONMENT_COMMAND");
      installCommandEnv.setValue(installCommand.toString());
      container.addEnvItem(installCommandEnv);
    }

    // Resources
    if (notebookPodSpec.getResources() != null) {
      V1ResourceRequirements resources = new V1ResourceRequirements();
      resources.setLimits(parseResources(notebookPodSpec));
      container.setResources(resources);
    }

    containers.add(container);
    podSpec.setContainers(containers);

    podTemplateSpec.setSpec(podSpec);
    return podTemplateSpec;
  }

  private static List<V1EnvVar> parseEnvVars(NotebookPodSpec podSpec) {
    if (podSpec.getEnvVars() == null)
      return null;
    List<V1EnvVar> envVars = new ArrayList<>();
    for (Map.Entry<String, String> entry : podSpec.getEnvVars().entrySet()) {
      V1EnvVar env = new V1EnvVar();
      env.setName(entry.getKey());
      env.setValue(entry.getValue());
      envVars.add(env);
    }
    return envVars;
  }

  private static Map<String, Quantity> parseResources(NotebookPodSpec podSpec) {

    Map<String, Quantity> resources = new HashMap<>();
    podSpec.setResources(podSpec.getResources());

    if (podSpec.getCpu() != null) {
      resources.put("cpu", new Quantity(podSpec.getCpu()));
    }
    if (podSpec.getMemory() != null) {
      resources.put("memory", new Quantity(podSpec.getMemory()));
    }
    if (podSpec.getGpu() != null) {
      resources.put("nvidia.com/gpu", new Quantity(podSpec.getGpu()));
    }
    return resources;
  }

  private static Environment getEnvironment(NotebookSpec notebookSpec) {
    if (notebookSpec.getEnvironment().getName() != null) {
      EnvironmentManager environmentManager = EnvironmentManager.getInstance();
      return environmentManager
              .getEnvironment(notebookSpec.getEnvironment().getName());
    } else {
      return null;
    }
  }

  private static String generateCondaVersionValidateCommand() {
    String currentVersion = "currentVersion=$(conda -V | cut -f2 -d' ');";
    String minVersion = "minVersion=\""
            + conf.getString(SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MIN_VERSION)
            + "\";";
    String maxVersion = "maxVersion=\""
            + conf.getString(SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MAX_VERSION)
            + "\";";
    StringBuffer condaVersionValidationCommand = new StringBuffer();
    condaVersionValidationCommand.append(minVersion);
    condaVersionValidationCommand.append(maxVersion);
    condaVersionValidationCommand.append(currentVersion);
    condaVersionValidationCommand.append("if [ \"$(printf '%s\\n' "
            + "\"$minVersion\" \"$maxVersion\" \"$currentVersion\" | sort -V "
            + "| head -n2 | tail -1 )\" != \"$currentVersion\" ]; then echo "
            + "\"Conda version should be between " + minVersion + " and "
            + maxVersion + "\"; exit 1; else echo \"Conda current version is "
            + currentVersion + ". Moving forward with env creation and "
            + "activation.\"; fi");
    return condaVersionValidationCommand.toString();
  }
}
