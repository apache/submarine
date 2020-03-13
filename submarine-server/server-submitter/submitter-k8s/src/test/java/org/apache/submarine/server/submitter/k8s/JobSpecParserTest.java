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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.spec.JobLibrarySpec;
import org.apache.submarine.server.api.spec.JobSpec;
import org.apache.submarine.server.api.spec.JobTaskSpec;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.apache.submarine.server.submitter.k8s.model.MLJobReplicaSpec;
import org.apache.submarine.server.submitter.k8s.model.MLJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobReplicaType;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobReplicaType;
import org.apache.submarine.server.submitter.k8s.parser.JobSpecParser;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public class JobSpecParserTest {

  private final String pytorchJobReqFile = "/pytorch_job_req.json";
  private final String tfJobReqFile = "/tf_mnist_req.json";

  @Test
  public void testValidTensorflowJobSpec() throws IOException,
      URISyntaxException, InvalidSpecException {
    JobSpec jobSpec = buildFromJsonFile(tfJobReqFile);
    TFJob tfJob = (TFJob) JobSpecParser.parseJob(jobSpec);
    // namespace
    String expectedNS = jobSpec.getSubmitterSpec().getNamespace();
    String actualNS = tfJob.getMetadata().getNamespace();
    Assert.assertEquals(expectedNS, actualNS);
    // job name
    String expectedJobName = jobSpec.getName();
    String actualJobName = tfJob.getMetadata().getName();
    Assert.assertEquals(expectedJobName, actualJobName);
    // framework name
    String expectedFramework = jobSpec.getLibrarySpec().getName().toLowerCase();
    String actualFrameworkName = JobLibrarySpec.
        SupportedMLFramework.TENSORFLOW.getName().toLowerCase();
    Assert.assertEquals(expectedFramework, actualFrameworkName);
    validateReplicaSpec(jobSpec, tfJob, TFJobReplicaType.Ps);
    validateReplicaSpec(jobSpec, tfJob, TFJobReplicaType.Worker);
  }

  @Test
  public void testInvalidTFJobSpec() throws IOException,
      URISyntaxException {
    JobSpec jobSpec = buildFromJsonFile(tfJobReqFile);
    // Case 1. Invalid framework name
    jobSpec.getLibrarySpec().setName("fooframework");
    try {
      JobSpecParser.parseJob(jobSpec);
      Assert.assertTrue("It should throw InvalidSpecException", false);
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid pytorch replica name. It can only be "master" and "worker"
    jobSpec = buildFromJsonFile(tfJobReqFile);
    jobSpec.getTaskSpecs().put("foo", jobSpec.getTaskSpecs().get(TFJobReplicaType.Ps));
    jobSpec.getTaskSpecs().remove(TFJobReplicaType.Ps);
    try {
      JobSpecParser.parseJob(jobSpec);
      Assert.assertTrue("It should throw InvalidSpecException", false);
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  @Test
  public void testValidPyTorchJobSpec() throws IOException,
      URISyntaxException, InvalidSpecException {
    JobSpec jobSpec = buildFromJsonFile(pytorchJobReqFile);
    PyTorchJob pyTorchJob = (PyTorchJob) JobSpecParser.parseJob(jobSpec);
    // namespace
    String expectedNS = jobSpec.getSubmitterSpec().getNamespace();
    String actualNS = pyTorchJob.getMetadata().getNamespace();
    Assert.assertEquals(expectedNS, actualNS);
    // job name
    String expectedJobName = jobSpec.getName();
    String actualJobName = pyTorchJob.getMetadata().getName();
    Assert.assertEquals(expectedJobName, actualJobName);
    // framework name
    String expectedFramework = jobSpec.getLibrarySpec().getName().toLowerCase();
    String actualFrameworkName = JobLibrarySpec.
        SupportedMLFramework.PYTORCH.getName().toLowerCase();
    Assert.assertEquals(expectedFramework, actualFrameworkName);
    validateReplicaSpec(jobSpec, pyTorchJob, PyTorchJobReplicaType.Master);
    validateReplicaSpec(jobSpec, pyTorchJob, PyTorchJobReplicaType.Worker);
  }

  @Test
  public void testInvalidPyTorchJobSpec() throws IOException,
      URISyntaxException {
    JobSpec jobSpec = buildFromJsonFile(pytorchJobReqFile);
    // Case 1. Invalid framework name
    jobSpec.getLibrarySpec().setName("fooframework");
    try {
      JobSpecParser.parseJob(jobSpec);
      Assert.assertTrue("It should throw InvalidSpecException", false);
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unsupported framework name"));
    }

    // Case 2. Invalid pytorch replica name. It can only be "master" and "worker"
    jobSpec = buildFromJsonFile(pytorchJobReqFile);
    jobSpec.getTaskSpecs().put("ps", jobSpec.getTaskSpecs().get(PyTorchJobReplicaType.Master));
    jobSpec.getTaskSpecs().remove(PyTorchJobReplicaType.Master);
    try {
      JobSpecParser.parseJob(jobSpec);
      Assert.assertTrue("It should throw InvalidSpecException", false);
    } catch (InvalidSpecException e) {
      Assert.assertTrue(e.getMessage().contains("Unrecognized replica type name"));
    }
  }

  public void validateReplicaSpec(JobSpec jobSpec,
      MLJob mlJob, MLJobReplicaType type) {
    MLJobReplicaSpec mlJobReplicaSpec = null;
    if (mlJob instanceof PyTorchJob) {
      mlJobReplicaSpec = ((PyTorchJob) mlJob).getSpec().getReplicaSpecs().get(type);
    } else if (mlJob instanceof TFJob){
      mlJobReplicaSpec = ((TFJob) mlJob).getSpec().getReplicaSpecs().get(type);
    }
    JobTaskSpec definedPyTorchMasterTask = jobSpec.getTaskSpecs().
        get(type.getTypeName());
    // replica
    int expectedMasterReplica = definedPyTorchMasterTask.getReplicas();
    Assert.assertEquals(expectedMasterReplica,
        (int) mlJobReplicaSpec.getReplicas());
    // Image
    String expectedMasterImage = definedPyTorchMasterTask.getImage() == null ?
        jobSpec.getLibrarySpec().getImage() : definedPyTorchMasterTask.getImage();
    String actualMasterImage = mlJobReplicaSpec.getContainerImageName();
    Assert.assertEquals(expectedMasterImage, actualMasterImage);
    // command
    String definedMasterCommandInTaskSpec = definedPyTorchMasterTask.getCmd();
    String expectedMasterCommand =
        definedMasterCommandInTaskSpec == null ?
            jobSpec.getLibrarySpec().getCmd() : definedMasterCommandInTaskSpec;
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

  private JobSpec buildFromJsonFile(String filePath) throws IOException,
      URISyntaxException {
    Gson gson = new GsonBuilder().create();
    try (Reader reader = Files.newBufferedReader(
        getCustomJobSpecFile(filePath).toPath(),
        StandardCharsets.UTF_8)) {
      return gson.fromJson(reader, JobSpec.class);
    }
  }

  private File getCustomJobSpecFile(String path) throws URISyntaxException {
    URL fileUrl = this.getClass().getResource(path);
    return new File(fileUrl.toURI());
  }
}
