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

package org.apache.submarine.server.submitter.k8s;

import java.io.IOException;
import java.net.URISyntaxException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1Volume;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.k8s.utils.K8sUtils;
import org.apache.submarine.server.manager.EnvironmentManager;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJob;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJobFactory;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJobReplicaSpec;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJob;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJobReplicaType;
import org.apache.submarine.server.submitter.k8s.experiment.codelocalizer.AbstractCodeLocalizer;
import org.apache.submarine.server.submitter.k8s.experiment.codelocalizer.GitCodeLocalizer;
import org.apache.submarine.server.submitter.k8s.experiment.codelocalizer.SSHGitCodeLocalizer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EmptyDirVolumeSource;
import io.kubernetes.client.openapi.models.V1EnvVar;

import static org.apache.submarine.commons.utils.SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MAX_VERSION;
import static org.apache.submarine.commons.utils.SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MIN_VERSION;


public class ExperimentSpecParserTest extends SpecBuilder {

  private static final SubmarineConfiguration conf =
      SubmarineConfiguration.getInstance();

  @Before
  public void beforeInit() {
    conf.setJdbcUrl(H2_JDBC_URL);
    conf.setJdbcDriverClassName(H2_JDBC_DRIVERCLASS);
    conf.setJdbcUserName(H2_JDBC_USERNAME);
    conf.setJdbcPassword(H2_JDBC_PASSWORD);
    try (Connection conn = DriverManager.getConnection(H2_JDBC_URL,
            H2_JDBC_USERNAME, H2_JDBC_PASSWORD);
         Statement stmt = conn.createStatement()) {
      stmt.execute("RUNSCRIPT FROM 'classpath:/db/experiment.sql'");
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  @Test
  public void testValidLabel() throws IOException, URISyntaxException {
    ExperimentSpec experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    String label = MLJobFactory.getJobLabelSelector(experimentSpec);
    Assert.assertEquals("training.kubeflow.org/job-name="
        + experimentSpec.getMeta().getExperimentId(), label);
  }

  @Test
  public void testValidTensorFlowExperiment() throws IOException,
      URISyntaxException, InvalidSpecException {
    ExperimentSpec experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    TFJob tfJob = (TFJob) MLJobFactory.getMLJob(experimentSpec);
    validateMetadata(experimentSpec.getMeta(), tfJob.getMetadata(),
        ExperimentMeta.SupportedMLFramework.TENSORFLOW.getName().toLowerCase()
    );
    // Validate ExperimentMeta without envVars. Related to SUBMARINE-534.
    experimentSpec.getMeta().setEnvVars(null);
    validateMetadata(experimentSpec.getMeta(), tfJob.getMetadata(),
            ExperimentMeta.SupportedMLFramework.TENSORFLOW.getName().toLowerCase()
    );

    validateReplicaSpec(experimentSpec, tfJob, TFJobReplicaType.Ps);
    validateReplicaSpec(experimentSpec, tfJob, TFJobReplicaType.Worker);
    validateExperimentHandlerMetadata(experimentSpec, tfJob);
  }

  @Test
  public void testInvalidTensorFlowExperiment() throws IOException,
      URISyntaxException {
    ExperimentSpec experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    // Case 1. Invalid framework name
    experimentSpec.getMeta().setFramework("fooframework");
    try {
      MLJobFactory.getMLJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid TensorFlow replica name. It can only be "ps" "worker" "chief" and "Evaluator"
    experimentSpec =  (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    experimentSpec.getSpec().put("foo", experimentSpec.getSpec().get(TFJobReplicaType.Ps.getTypeName()));
    experimentSpec.getSpec().remove(TFJobReplicaType.Ps.getTypeName());
    try {
      MLJobFactory.getMLJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  @Test
  public void testValidPyTorchExperiment() throws IOException,
      URISyntaxException, InvalidSpecException {
    ExperimentSpec experimentSpec =
            (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, pytorchJobReqFile);
    PyTorchJob pyTorchJob = (PyTorchJob) MLJobFactory.getMLJob(experimentSpec);
    validateMetadata(experimentSpec.getMeta(), pyTorchJob.getMetadata(),
        ExperimentMeta.SupportedMLFramework.PYTORCH.getName().toLowerCase()
    );
    // Validate ExperimentMeta without envVars. Related to SUBMARINE-534.
    experimentSpec.getMeta().setEnvVars(null);
    validateMetadata(experimentSpec.getMeta(), pyTorchJob.getMetadata(),
            ExperimentMeta.SupportedMLFramework.PYTORCH.getName().toLowerCase()
    );

    validateReplicaSpec(experimentSpec, pyTorchJob, PyTorchJobReplicaType.Master);
    validateReplicaSpec(experimentSpec, pyTorchJob, PyTorchJobReplicaType.Worker);
    validateExperimentHandlerMetadata(experimentSpec, pyTorchJob);
  }

  @Test
  public void testInvalidPyTorchJobSpec() throws IOException,
      URISyntaxException {
    ExperimentSpec experimentSpec =
            (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, pytorchJobReqFile);
    // Case 1. Invalid framework name
    experimentSpec.getMeta().setFramework("fooframework");
    try {
      MLJobFactory.getMLJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid PyTorch replica name. It can only be "master" and "worker"
    experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, pytorchJobReqFile);
    experimentSpec.getSpec().put("ps", experimentSpec.getSpec().get(
        PyTorchJobReplicaType.Master.getTypeName()));
    experimentSpec.getSpec().remove(PyTorchJobReplicaType.Master.getTypeName());
    try {
      MLJobFactory.getMLJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  @Test
  public void testValidXGBoostExperiment() throws IOException,
      URISyntaxException, InvalidSpecException {
    ExperimentSpec experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class,
        xgboostJobReqFile);
    XGBoostJob xgboostJob = (XGBoostJob) MLJobFactory.getMLJob(experimentSpec);
    validateMetadata(experimentSpec.getMeta(), xgboostJob.getMetadata(),
        ExperimentMeta.SupportedMLFramework.XGBOOST.getName().toLowerCase()
    );

    validateReplicaSpec(experimentSpec, xgboostJob, XGBoostJobReplicaType.Master);
    validateReplicaSpec(experimentSpec, xgboostJob, XGBoostJobReplicaType.Worker);
    validateExperimentHandlerMetadata(experimentSpec, xgboostJob);
  }

  @Test
  public void testInvalidXGBoostExperiment() throws IOException,
      URISyntaxException {
    ExperimentSpec experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class,
        xgboostJobReqFile);
    // Case 1. Invalid framework name
    experimentSpec.getMeta().setFramework("fooframework");
    try {
      MLJobFactory.getMLJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid XGBoost Replica name. It can only be "master" and "worker"
    experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, xgboostJobReqFile);
    experimentSpec.getSpec().put("foo", experimentSpec.getSpec().get(
        XGBoostJobReplicaType.Master.getTypeName()));
    experimentSpec.getSpec().remove(XGBoostJobReplicaType.Master.getTypeName());
    try {
      MLJobFactory.getMLJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  private void validateMetadata(ExperimentMeta expectedMeta, V1ObjectMeta actualMeta,
      String actualFramework) {
    Assert.assertEquals(expectedMeta.getExperimentId(), actualMeta.getName());
    Assert.assertEquals(K8sUtils.getNamespace(), actualMeta.getNamespace());
    Assert.assertEquals(expectedMeta.getFramework().toLowerCase(), actualFramework);
  }

  private void validateExperimentHandlerMetadata(ExperimentSpec experimentSpec,
      MLJob mlJob) {

    if (experimentSpec.getExperimentHandlerSpec() == null ||
        experimentSpec.getExperimentHandlerSpec().isEmpty()) {
      return;
    }

    V1Container initContainer = null;

    MLJobReplicaSpec mlJobReplicaSpec = null;
    if (mlJob instanceof PyTorchJob) {
      mlJobReplicaSpec = ((PyTorchJob) mlJob).getSpec()
        .getReplicaSpecs().get(PyTorchJobReplicaType.Master);
    } else if (mlJob instanceof TFJob) {
      mlJobReplicaSpec = ((TFJob) mlJob).getSpec()
        .getReplicaSpecs().get(TFJobReplicaType.Ps);
    } else if (mlJob instanceof XGBoostJob) {
      mlJobReplicaSpec = ((XGBoostJob) mlJob).getSpec()
        .getReplicaSpecs().get(XGBoostJobReplicaType.Master);
    }
    initContainer = mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().get(0);
    Map<String, String> varMap = initContainer.getEnv().stream()
        .collect(Collectors.toMap(V1EnvVar::getName, V1EnvVar::getValue));
    Assert.assertEquals(experimentSpec.getExperimentHandlerSpec().get("FILE_SYSTEM_TYPE")
        , varMap.get("FILE_SYSTEM_TYPE"));
    Assert.assertEquals(experimentSpec.getExperimentHandlerSpec().get("HDFS_HOST")
        , varMap.get("HDFS_HOST"));
    Assert.assertEquals(experimentSpec.getExperimentHandlerSpec().get("HDFS_PORT")
        , varMap.get("HDFS_PORT"));
    Assert.assertEquals(experimentSpec.getExperimentHandlerSpec().get("HDFS_SOURCE")
        , varMap.get("HDFS_SOURCE"));
    Assert.assertEquals(experimentSpec.getExperimentHandlerSpec().get("ENABLE_KERBEROS")
        , varMap.get("ENABLE_KERBEROS"));
    Assert.assertEquals(mlJob.getExperimentId(), varMap.get("EXPERIMENT_ID"));
  }

  private void validateReplicaSpec(ExperimentSpec experimentSpec,
      MLJob mlJob, MLJobReplicaType type) {
    MLJobReplicaSpec mlJobReplicaSpec = null;
    if (mlJob instanceof PyTorchJob) {
      mlJobReplicaSpec = ((PyTorchJob) mlJob).getSpec().getReplicaSpecs().get(type);
    } else if (mlJob instanceof TFJob) {
      mlJobReplicaSpec = ((TFJob) mlJob).getSpec().getReplicaSpecs().get(type);
    } else if (mlJob instanceof XGBoostJob) {
      mlJobReplicaSpec = ((XGBoostJob) mlJob).getSpec().getReplicaSpecs().get(type);
    }
    Assert.assertNotNull(mlJobReplicaSpec);

    ExperimentTaskSpec definedPyTorchMasterTask = experimentSpec.getSpec().
        get(type.getTypeName());

    // replica
    int expectedMasterReplica = definedPyTorchMasterTask.getReplicas();
    Assert.assertEquals(expectedMasterReplica,
        (int) mlJobReplicaSpec.getReplicas());
    // image
    String expectedMasterImage = definedPyTorchMasterTask.getImage() == null ?
        experimentSpec.getEnvironment().getImage() : definedPyTorchMasterTask.getImage();
    String actualMasterImage = mlJobReplicaSpec.getContainerImageName();
    Assert.assertEquals(expectedMasterImage, actualMasterImage);
    // command
    String definedMasterCommandInTaskSpec = definedPyTorchMasterTask.getCmd();
    String expectedMasterCommand =
        definedMasterCommandInTaskSpec == null ?
            experimentSpec.getMeta().getCmd() : definedMasterCommandInTaskSpec;
    String actualMasterContainerCommand = mlJobReplicaSpec.getContainerCommand();
    Assert.assertEquals(expectedMasterCommand,
        actualMasterContainerCommand);
    // mem
    String expectedMasterContainerMem = definedPyTorchMasterTask.getMemory();
    String actualMasterContainerMem = mlJobReplicaSpec.getContainerMemMB();
    Assert.assertEquals(expectedMasterContainerMem,
        actualMasterContainerMem);

    // cpu
    String expectedMasterContainerCpu = definedPyTorchMasterTask.getCpu();
    String actualMasterContainerCpu = mlJobReplicaSpec.getContainerCpu();
    Assert.assertEquals(expectedMasterContainerCpu,
        actualMasterContainerCpu);
  }

  @Test
  public void testValidPyTorchJobSpecWithEnv()
      throws IOException, URISyntaxException, InvalidSpecException {

    EnvironmentManager environmentManager = EnvironmentManager.getInstance();

    EnvironmentSpec spec = new EnvironmentSpec();
    String envName = "my-submarine-env";
    spec.setName(envName);
    String dockerImage = "example.com/my-docker-image:0.1.2";
    spec.setDockerImage(dockerImage);

    KernelSpec kernelSpec = new KernelSpec();
    String kernelName = "team_python";
    kernelSpec.setName(kernelName);
    List<String> channels = new ArrayList<String>();
    String channel = "default";
    channels.add(channel);
    kernelSpec.setChannels(channels);

    List<String> dependencies = new ArrayList<String>();
    String dependency = "_ipyw_jlab_nb_ext_conf=0.1.0=py37_0";
    dependencies.add(dependency);
    kernelSpec.setCondaDependencies(dependencies);
    spec.setKernelSpec(kernelSpec);

    environmentManager.createEnvironment(spec);

    ExperimentSpec jobSpec =
            (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, pytorchJobWithEnvReqFile);
    PyTorchJob pyTorchJob = (PyTorchJob) MLJobFactory.getMLJob(jobSpec);

    MLJobReplicaSpec mlJobReplicaSpec = pyTorchJob.getSpec().getReplicaSpecs()
        .get(PyTorchJobReplicaType.Master);
    Assert.assertEquals(1,
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().size());
    V1Container initContainer =
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().get(0);
    Assert.assertEquals(dockerImage, initContainer.getImage());

    Assert.assertEquals("/bin/bash", initContainer.getCommand().get(0));
    Assert.assertEquals("-c", initContainer.getCommand().get(1));

    String minVersion = "minVersion=\""
        + conf.getString(
            ENVIRONMENT_CONDA_MIN_VERSION)
        + "\";";
    String maxVersion = "maxVersion=\""
        + conf.getString(
            ENVIRONMENT_CONDA_MAX_VERSION)
        + "\";";
    String currentVersion = "currentVersion=$(conda -V | cut -f2 -d' ');";
    Assert.assertEquals(
        minVersion + maxVersion + currentVersion
            + "if [ \"$(printf '%s\\n' \"$minVersion\" \"$maxVersion\" "
               + "\"$currentVersion\" | sort -V | head -n2 | tail -1 )\" "
                    + "!= \"$currentVersion\" ]; then echo \"Conda version " +
                    "should be between minVersion=\""
                    + ENVIRONMENT_CONDA_MIN_VERSION.getStringValue() + "\"; "
                    + "and maxVersion=\"" + ENVIRONMENT_CONDA_MAX_VERSION.getStringValue()
                    + "\";\"; exit 1; else echo "
                    + "\"Conda current version is " + currentVersion + ". "
                        + "Moving forward with env creation and activation.\"; "
                        + "fi && " +
        "conda create -n " + kernelName + " -c " + channel + " " + dependency
            + " && " + "echo \"source activate " + kernelName + "\" > ~/.bashrc"
            + " && " + "PATH=/opt/conda/envs/env/bin:$PATH",
        initContainer.getCommand().get(2));

    environmentManager.deleteEnvironment(envName);
  }

  @Test
  public void testValidPyTorchJobSpecWithHTTPGitCodeLocalizer()
      throws IOException, URISyntaxException, InvalidSpecException {
    ExperimentSpec jobSpec =
        (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class,
            pytorchJobWithHTTPGitCodeLocalizerFile);
    PyTorchJob pyTorchJob = (PyTorchJob) MLJobFactory.getMLJob(jobSpec);

    MLJobReplicaSpec mlJobReplicaSpec = pyTorchJob.getSpec().getReplicaSpecs()
        .get(PyTorchJobReplicaType.Master);
    Assert.assertEquals(1,
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().size());
    V1Container initContainer =
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().get(0);
    Assert.assertEquals(
        AbstractCodeLocalizer.CODE_LOCALIZER_INIT_CONTAINER_NAME,
        initContainer.getName());
    Assert.assertEquals(GitCodeLocalizer.GIT_SYNC_IMAGE,
        initContainer.getImage());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME,
        initContainer.getVolumeMounts().get(0).getName());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH,
        initContainer.getVolumeMounts().get(0).getMountPath());

    V1Container container =
        mlJobReplicaSpec.getTemplate().getSpec().getContainers().get(0);
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME,
        container.getVolumeMounts().get(0).getName());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH,
        container.getVolumeMounts().get(0).getMountPath());
    for (V1EnvVar env : container.getEnv()) {
      if (env.getName()
          .equals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH_ENV_VAR)) {
        Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH,
            env.getValue());
      }
    }

    // we need to filter code-dir first
    V1Volume V1Volume = mlJobReplicaSpec.getTemplate().getSpec().getVolumes().stream()
        .filter(v -> v.getName().equals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME))
        .findFirst().get();
    Assert.assertEquals(new V1EmptyDirVolumeSource(), V1Volume.getEmptyDir());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME,
        V1Volume.getName());
  }

  @Test
  public void testValidPyTorchJobSpecWithSSHGitCodeLocalizer()
      throws IOException, URISyntaxException, InvalidSpecException {
    ExperimentSpec jobSpec =
        (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class,
            pytorchJobWithSSHGitCodeLocalizerFile);
    PyTorchJob pyTorchJob = (PyTorchJob) MLJobFactory.getMLJob(jobSpec);

    MLJobReplicaSpec mlJobReplicaSpec = pyTorchJob.getSpec().getReplicaSpecs()
        .get(PyTorchJobReplicaType.Master);
    Assert.assertEquals(1,
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().size());
    V1Container initContainer =
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().get(0);
    Assert.assertEquals(
        AbstractCodeLocalizer.CODE_LOCALIZER_INIT_CONTAINER_NAME,
        initContainer.getName());
    Assert.assertEquals(GitCodeLocalizer.GIT_SYNC_IMAGE,
        initContainer.getImage());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME,
        initContainer.getVolumeMounts().get(0).getName());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH,
        initContainer.getVolumeMounts().get(0).getMountPath());
    for (V1EnvVar env : initContainer.getEnv()) {
      if (env.getName().equals(SSHGitCodeLocalizer.GIT_SYNC_SSH_NAME)) {
        Assert.assertEquals(SSHGitCodeLocalizer.GIT_SYNC_SSH_VALUE,
            env.getValue());
      }
    }

    V1Container container =
        mlJobReplicaSpec.getTemplate().getSpec().getInitContainers().get(0);
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME,
        container.getVolumeMounts().get(0).getName());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH,
        container.getVolumeMounts().get(0).getMountPath());
    for (V1EnvVar env : container.getEnv()) {
      if (env.getName()
          .equals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH_ENV_VAR)) {
        Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_PATH,
            env.getValue());
      }
    }

    // we need to filter code-dir first
    V1Volume V1Volume = mlJobReplicaSpec.getTemplate().getSpec().getVolumes().stream()
        .filter(v -> v.getName().equals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME))
        .findFirst().get();
    Assert.assertEquals(new V1EmptyDirVolumeSource(), V1Volume.getEmptyDir());
    Assert.assertEquals(AbstractCodeLocalizer.CODE_LOCALIZER_MOUNT_NAME,
        V1Volume.getName());
  }
}
