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

import io.kubernetes.client.models.V1ObjectMeta;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.apache.submarine.server.submitter.k8s.model.MLJobReplicaSpec;
import org.apache.submarine.server.submitter.k8s.model.MLJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobReplicaType;
import org.apache.submarine.server.submitter.k8s.parser.ExperimentSpecParser;
import org.junit.Assert;
import org.junit.Test;

public class ExperimentSpecParserTest extends SpecBuilder {
  @Test
  public void testValidTensorFlowExperiment() throws IOException,
      URISyntaxException, InvalidSpecException {
    ExperimentSpec experimentSpec = buildFromJsonFile(tfJobReqFile);
    TFJob tfJob = (TFJob) ExperimentSpecParser.parseJob(experimentSpec);
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
  }

  @Test
  public void testInvalidTensorFlowExperiment() throws IOException,
      URISyntaxException {
    ExperimentSpec experimentSpec = buildFromJsonFile(tfJobReqFile);
    // Case 1. Invalid framework name
    experimentSpec.getMeta().setFramework("fooframework");
    try {
      ExperimentSpecParser.parseJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid TensorFlow replica name. It can only be "ps" "worker" "chief" and "Evaluator"
    experimentSpec = buildFromJsonFile(tfJobReqFile);
    experimentSpec.getSpec().put("foo", experimentSpec.getSpec().get(TFJobReplicaType.Ps.getTypeName()));
    experimentSpec.getSpec().remove(TFJobReplicaType.Ps.getTypeName());
    try {
      ExperimentSpecParser.parseJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  @Test
  public void testValidPyTorchExperiment() throws IOException,
      URISyntaxException, InvalidSpecException {
    ExperimentSpec experimentSpec = buildFromJsonFile(pytorchJobReqFile);
    PyTorchJob pyTorchJob = (PyTorchJob) ExperimentSpecParser.parseJob(experimentSpec);
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
  }

  @Test
  public void testInvalidPyTorchJobSpec() throws IOException,
      URISyntaxException {
    ExperimentSpec experimentSpec = buildFromJsonFile(pytorchJobReqFile);
    // Case 1. Invalid framework name
    experimentSpec.getMeta().setFramework("fooframework");
    try {
      ExperimentSpecParser.parseJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid PyTorch replica name. It can only be "master" and "worker"
    experimentSpec = buildFromJsonFile(pytorchJobReqFile);
    experimentSpec.getSpec().put("ps", experimentSpec.getSpec().get(
        PyTorchJobReplicaType.Master.getTypeName()));
    experimentSpec.getSpec().remove(PyTorchJobReplicaType.Master.getTypeName());
    try {
      ExperimentSpecParser.parseJob(experimentSpec);
      Assert.fail("It should throw InvalidSpecException");
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  private void validateMetadata(ExperimentMeta expectedMeta, V1ObjectMeta actualMeta,
      String actualFramework) {
    Assert.assertEquals(expectedMeta.getName(), actualMeta.getName());
    Assert.assertEquals(expectedMeta.getNamespace(), actualMeta.getNamespace());
    Assert.assertEquals(expectedMeta.getFramework().toLowerCase(), actualFramework);
  }

  private void validateReplicaSpec(ExperimentSpec experimentSpec,
      MLJob mlJob, MLJobReplicaType type) {
    MLJobReplicaSpec mlJobReplicaSpec = null;
    if (mlJob instanceof PyTorchJob) {
      mlJobReplicaSpec = ((PyTorchJob) mlJob).getSpec().getReplicaSpecs().get(type);
    } else if (mlJob instanceof TFJob){
      mlJobReplicaSpec = ((TFJob) mlJob).getSpec().getReplicaSpecs().get(type);
    }
    Assert.assertNotNull(mlJobReplicaSpec);

    ExperimentTaskSpec definedPyTorchMasterTask = experimentSpec.getSpec().
        get(type.getTypeName());
    // replica
    int expectedMasterReplica = definedPyTorchMasterTask.getReplicas();
    Assert.assertEquals(expectedMasterReplica,
        (int) mlJobReplicaSpec.getReplicas());
    // Image
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
}
