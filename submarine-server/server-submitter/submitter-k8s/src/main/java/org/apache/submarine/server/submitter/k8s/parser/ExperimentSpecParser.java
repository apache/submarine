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

import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1EnvVarSource;
import io.kubernetes.client.openapi.models.V1ObjectMetaBuilder;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaimVolumeSource;
import io.kubernetes.client.openapi.models.V1PodSecurityContext;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;
import io.kubernetes.client.openapi.models.V1ResourceRequirements;
import io.kubernetes.client.openapi.models.V1SecretKeySelector;
import io.kubernetes.client.openapi.models.V1SecretVolumeSource;
import io.kubernetes.client.openapi.models.V1Volume;
import io.kubernetes.client.openapi.models.V1VolumeMount;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.manager.EnvironmentManager;
import org.apache.submarine.server.s3.S3Constants;
import org.apache.submarine.server.submitter.k8s.experiment.codelocalizer.AbstractCodeLocalizer;
import org.apache.submarine.server.submitter.k8s.experiment.codelocalizer.CodeLocalizer;
import org.apache.submarine.server.submitter.k8s.experiment.codelocalizer.SSHGitCodeLocalizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ExperimentSpecParser {

  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  /**
   * S3 secret
   */
  public static final String MLFLOW_S3_ENDPOINT_URL = "MLFLOW_S3_ENDPOINT_URL";
  public static final String AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID";
  public static final String AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY";
  public static final String MINIO_ACCESS_KEY = "MINIO_ACCESS_KEY";
  public static final String MINIO_SECRET_KEY = "MINIO_SECRET_KEY";
  public static final String DEFAULT_SUBMARINE_MINIO_SECRET = "submarine-minio-secret";

  public static V1PodTemplateSpec parseTemplateSpec(
      ExperimentTaskSpec taskSpec, ExperimentSpec experimentSpec) throws InvalidSpecException {
    V1PodTemplateSpec templateSpec = new V1PodTemplateSpec();
    // There is no need to add istio sidecar. Otherwise, the pod may not end normally
    // https://istio.io/latest/docs/setup/additional-setup/sidecar-injection/
    // Controlling the injection policy Section
    V1ObjectMetaBuilder metaBuilder = new V1ObjectMetaBuilder()
            .addToAnnotations("sidecar.istio.io/inject", "false");
    templateSpec.setMetadata(metaBuilder.build());
    V1PodSpec podSpec = new V1PodSpec();
    List<V1Container> containers = new ArrayList<>();
    V1Container container = new V1Container();
    container.setName(experimentSpec.getMeta().getFramework().toLowerCase());
    // image
    if (taskSpec.getImage() != null) {
      container.setImage(taskSpec.getImage());
    } else {
      if (experimentSpec.getEnvironment().getImage() != null) {
        container.setImage(experimentSpec.getEnvironment().getImage());
      }
    }
    // cmd
    if (taskSpec.getCmd() != null) {
      container.setCommand(Arrays.asList(taskSpec.getCmd().split(" ")));
    } else {
      container.setCommand(Arrays.asList(experimentSpec.getMeta().getCmd().split(" ")));
    }
    // resources
    V1ResourceRequirements resources = new V1ResourceRequirements();
    resources.setRequests(parseResources(taskSpec, true));
    resources.setLimits(parseResources(taskSpec, false));
    container.setResources(resources);
    container.setEnv(parseEnvVars(taskSpec, experimentSpec.getMeta().getEnvVars()));


    final String name = experimentSpec.getMeta().getName();
    // volumeMount
    container.addVolumeMountsItem(
        new V1VolumeMount().mountPath("/logs").
        name("volume").subPath("submarine-tensorboard/" + name)
    );

    // volume
    V1Volume podVolume = new V1Volume().name("volume");
    podVolume.setPersistentVolumeClaim(
        new V1PersistentVolumeClaimVolumeSource().claimName("submarine-tensorboard-pvc")
    );
    podSpec.addVolumesItem(podVolume);
    /**
     * Init Git localize Container
     */
    if (experimentSpec.getCode() != null) {
      CodeLocalizer localizer = AbstractCodeLocalizer.getCodeLocalizer(
          experimentSpec.getCode());
      localizer.localize(podSpec);

      if (podSpec.getInitContainers() != null
          && podSpec.getInitContainers().size() > 0) {

        List<V1VolumeMount> initContainerVolumeMounts =
            podSpec.getInitContainers().get(0).getVolumeMounts();
        List<V1VolumeMount> volumeMounts = new ArrayList<V1VolumeMount>();

        // Populate container volume mounts using Init container info
        for (V1VolumeMount initContainerVolumeMount : initContainerVolumeMounts) {
          String volumeName = initContainerVolumeMount.getName();
          String path = initContainerVolumeMount.getMountPath();
          if (volumeName
              .equals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME)) {
            V1VolumeMount mount = new V1VolumeMount();
            mount.setName(volumeName);
            mount.setMountPath(path);
            volumeMounts.add(mount);
            container.setVolumeMounts(volumeMounts);
          } else if (volumeName
              .equals(SSHGitCodeLocalizer.GIT_SECRET_MOUNT_NAME)) {
            V1Volume volume = new V1Volume();
            volume.setName(volumeName);

            List<V1Volume> existingVolumes = podSpec.getVolumes();
            V1SecretVolumeSource secret = new V1SecretVolumeSource();
            secret.secretName(SSHGitCodeLocalizer.GIT_SECRET_NAME);
            secret.setDefaultMode(SSHGitCodeLocalizer.GIT_SECRET_MODE);
            volume.setSecret(secret);
            existingVolumes.add(volume);

            // Pod level security context
            V1PodSecurityContext podSecurityContext =
                new V1PodSecurityContext();
            podSecurityContext.setFsGroup(SSHGitCodeLocalizer.GIT_SYNC_USER);
            podSpec.setSecurityContext(podSecurityContext);
          }
        }

        V1EnvVar codeEnvVar = new V1EnvVar();
        codeEnvVar.setName(AbstractCodeLocalizer.CODE_LOCALIZER_PATH_ENV_VAR);
        codeEnvVar.setValue(AbstractCodeLocalizer.CODE_LOCALIZER_PATH);
        List<V1EnvVar> envVars = container.getEnv();
        envVars.add(codeEnvVar);
        container.setEnv(envVars);
      }
    }

    containers.add(container);
    podSpec.setContainers(containers);

    /**
     * Init Containers
     */
    if (experimentSpec.getEnvironment().getName() != null) {
      Environment environment = getEnvironment(experimentSpec);
      if (environment != null) {
        EnvironmentSpec environmentSpec = environment.getEnvironmentSpec();
        List<V1Container> initContainers = new ArrayList<>();
        V1Container initContainer = new V1Container();
        initContainer.setName(environment.getEnvironmentSpec().getName());
        initContainer.setImage(environmentSpec.getDockerImage());

        if (environmentSpec.getKernelSpec().getCondaDependencies().size() > 0) {

          String minVersion = "minVersion=\""
              + conf.getString(
                  SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MIN_VERSION)
              + "\";";
          String maxVersion = "maxVersion=\""
              + conf.getString(
                  SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MAX_VERSION)
              + "\";";
          String currentVersion = "currentVersion=$(conda -V | cut -f2 -d' ');";
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

          StringBuffer createCommand = new StringBuffer();
          String condaEnvironmentName =
              environmentSpec.getKernelSpec().getName();

          createCommand.append("conda create -n " + condaEnvironmentName);
          for (String channel : environmentSpec.getKernelSpec().getChannels()) {
            createCommand.append(" ");
            createCommand.append("-c");
            createCommand.append(" ");
            createCommand.append(channel);
          }
          for (String dependency : environmentSpec.getKernelSpec()
              .getCondaDependencies()) {
            createCommand.append(" ");
            createCommand.append(dependency);
          }
          String activateCommand = "echo \"source activate "
              + condaEnvironmentName + "\" > ~/.bashrc";
          String pathCommand = "PATH=/opt/conda/envs/env/bin:$PATH";
          String finalCommand = condaVersionValidationCommand.toString() +
              " && " + createCommand.toString() + " && "
              + activateCommand + " && " + pathCommand;
          initContainer.addCommandItem("/bin/bash");
          initContainer.addCommandItem("-c");
          initContainer.addCommandItem(finalCommand);
        }
        initContainers.add(initContainer);
        podSpec.setInitContainers(initContainers);
      }
    }
    templateSpec.setSpec(podSpec);
    return templateSpec;
  }

  private static List<V1EnvVar> parseEnvVars(ExperimentTaskSpec spec,
      Map<String, String> defaultEnvs) {
    if (spec.getEnvVars() != null) {
      return parseEnvVars(spec.getEnvVars());
    }
    return parseEnvVars(defaultEnvs);
  }

  private static List<V1EnvVar> parseEnvVars(Map<String, String> envMap) {
    if (envMap == null)
      return null;
    List<V1EnvVar> envVars = new ArrayList<>();
    for (Map.Entry<String, String> entry : envMap.entrySet()) {
      V1EnvVar env = new V1EnvVar();
      env.setName(entry.getKey());
      env.setValue(entry.getValue());
      envVars.add(env);
    }
    // add s3(minio) url and secrets
    envVars.add(
        new V1EnvVar().name(MLFLOW_S3_ENDPOINT_URL).value(S3Constants.DEFAULT_ENDPOINT)
    );
    envVars.add(new V1EnvVar().name(AWS_ACCESS_KEY_ID)
        .valueFrom(new V1EnvVarSource()
            .secretKeyRef(new V1SecretKeySelector()
                .key(MINIO_ACCESS_KEY)
                .name(DEFAULT_SUBMARINE_MINIO_SECRET)
            )
        )
    );
    envVars.add(new V1EnvVar().name(AWS_SECRET_ACCESS_KEY)
        .valueFrom(new V1EnvVarSource()
            .secretKeyRef(new V1SecretKeySelector()
                .key(MINIO_SECRET_KEY)
                .name(DEFAULT_SUBMARINE_MINIO_SECRET)
            )
        )
    );
    return envVars;
  }

  private static Map<String, Quantity> parseResources(ExperimentTaskSpec taskSpec, boolean request) {
    Map<String, Quantity> resources = new HashMap<>();
    taskSpec.setResources(taskSpec.getResources());
    if (taskSpec.getCpu() != null) {
      resources.put("cpu", new Quantity(taskSpec.getCpu()));
    }
    if (taskSpec.getMemory() != null) {
      String memoryRequest = taskSpec.getMemory();
      if (request) {
        resources.put("memory", new Quantity(memoryRequest)); // ex: 1024M
      } else {
        // SUBMARINE-948. Allow experiments to overcommit memory
        String suffix = memoryRequest.substring(memoryRequest.length() - 1);
        String value = memoryRequest.substring(0, memoryRequest.length() - 1);
        String memoryLimit = String.valueOf(Integer.parseInt(value) * 2) + suffix;
        resources.put("memory", new Quantity(memoryLimit));
      }
    }
    if (taskSpec.getGpu() != null) {
      resources.put("nvidia.com/gpu", new Quantity(taskSpec.getGpu()));
    }
    return resources;
  }

  private static Environment getEnvironment(ExperimentSpec experimentSpec) {
    if (experimentSpec.getEnvironment().getName() != null) {
      EnvironmentManager environmentManager = EnvironmentManager.getInstance();
      return environmentManager
          .getEnvironment(experimentSpec.getEnvironment().getName());
    } else {
      return null;
    }
  }
}
