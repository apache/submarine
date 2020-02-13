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

package org.apache.submarine.client.cli.runjob.mxnet;

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.util.resource.Resources;
import org.apache.submarine.client.cli.param.runjob.MXNetRunJobParameters;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.client.cli.runjob.RunJobCli;
import org.apache.submarine.client.cli.runjob.RunJobCliParsingCommonTest;
import org.apache.submarine.commons.runtime.conf.SubmarineLogs;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.*;
import static org.junit.Assert.assertTrue;

/**
 * Test class that verifies the correctness of MXNet
 * CLI configuration parsing.
 */
public class RunJobCliParsingMXNetTest {
  @Before
  public void before() {
    SubmarineLogs.verboseOff();
  }

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Test
  public void testBasicRunJobForSingleNodeTraining() throws Exception {
    RunJobCli runJobCli = new RunJobCli(RunJobCliParsingCommonTest.getMockClientContext());
    assertFalse(SubmarineLogs.isVerbose());

    runJobCli.run(
        new String[]{"--framework", "mxnet", "--name", "my-job",
            "--docker_image", "mxnet-docker:1.1.0",
            "--input_path", "hdfs://input",
            "--num_workers", "1", "--num_ps", "1", "--worker_launch_cmd",
            "python run-job.py", "--worker_resources", "memory=2048M,vcores=2",
            "--ps_resources", "memory=4G,vcores=2", "--ps_launch_cmd",
            "python run-ps.py", "--num_schedulers", "1", "--scheduler_launch_cmd",
            "python run-scheduler.py", "--scheduler_resources", "memory=1024M,vcores=2",
            "--verbose", "--wait_job_finish"});

    RunJobParameters jobRunParameters = runJobCli.getRunJobParameters();
    assertTrue(RunJobParameters.class +
        " must be an instance of " +
        MXNetRunJobParameters.class,
        jobRunParameters instanceof MXNetRunJobParameters);
    MXNetRunJobParameters mxNetParams =
        (MXNetRunJobParameters) jobRunParameters;

    assertEquals(jobRunParameters.getInputPath(), "hdfs://input");
    assertEquals(mxNetParams.getNumWorkers(), 1);
    assertEquals(mxNetParams.getWorkerLaunchCmd(), "python run-job.py");
    assertEquals(Resources.createResource(2048, 2),
        mxNetParams.getWorkerResource());
    assertEquals(mxNetParams.getNumPS(), 1);
    assertEquals(mxNetParams.getNumSchedulers(), 1);
    assertTrue(SubmarineLogs.isVerbose());
    assertTrue(jobRunParameters.isWaitJobFinish());
  }

  @Test
  public void testBasicRunJobForDistributedTraining() throws Exception {
    RunJobCli runJobCli = new RunJobCli(RunJobCliParsingCommonTest.getMockClientContext());
    assertFalse(SubmarineLogs.isVerbose());
    runJobCli.run(
        new String[]{"--framework", "mxnet", "--name", "my-job",
            "--docker_image", "mxnet-docker:1.1.0",
            "--input_path", "hdfs://input",
            "--num_workers", "2", "--num_ps", "2", "--worker_launch_cmd",
            "python run-job.py", "--worker_resources", "memory=2048M,vcores=2",
            "--ps_resources", "memory=4G,vcores=2", "--ps_launch_cmd",
            "python run-ps.py", "--num_schedulers", "1", "--scheduler_launch_cmd",
            "python run-scheduler.py", "--scheduler_resources", "memory=1024M,vcores=2",
            "--verbose"});
    RunJobParameters jobRunParameters = runJobCli.getRunJobParameters();
    assertTrue(RunJobParameters.class +
        " must be an instance of " +
        MXNetRunJobParameters.class,
        jobRunParameters instanceof MXNetRunJobParameters);
    MXNetRunJobParameters mxNetParams =
        (MXNetRunJobParameters) jobRunParameters;

    assertEquals(jobRunParameters.getInputPath(), "hdfs://input");
    assertEquals(jobRunParameters.getDockerImageName(), "mxnet-docker:1.1.0");
    assertEquals(mxNetParams.getNumWorkers(), 2);
    assertEquals(Resources.createResource(2048, 2),
        mxNetParams.getWorkerResource());
    assertEquals(mxNetParams.getWorkerLaunchCmd(), "python run-job.py");
    assertEquals(mxNetParams.getNumPS(), 2);
    assertEquals(Resources.createResource(4096, 2),
        mxNetParams.getPsResource());
    assertEquals(mxNetParams.getPSLaunchCmd(), "python run-ps.py");
    assertEquals(mxNetParams.getNumSchedulers(), 1);
    assertEquals(Resources.createResource(1024, 2),
        mxNetParams.getSchedulerResource());
    assertEquals(mxNetParams.getSchedulerLaunchCmd(), "python run-scheduler.py");
    assertTrue(SubmarineLogs.isVerbose());
  }

  @Test
  public void testTensorboardCannotBeDefined() throws Exception {
    RunJobCli runJobCli = new RunJobCli(RunJobCliParsingCommonTest.getMockClientContext());
    assertFalse(SubmarineLogs.isVerbose());

    expectedException.expect(ParseException.class);
    expectedException.expectMessage("cannot be defined for MXNet jobs");
    runJobCli.run(
        new String[]{"--framework", "mxnet",
            "--name", "my-job", "--docker_image", "mxnet-docker:1.1.0",
            "--input_path", "hdfs://input",
            "--num_workers", "2",
            "--worker_launch_cmd", "python run-job.py",
            "--worker_resources", "memory=2048M,vcores=2",
            "--tensorboard"});
  }

  @Test
  public void testTensorboardResourcesCannotBeDefined() throws Exception {
    RunJobCli runJobCli = new RunJobCli(RunJobCliParsingCommonTest.getMockClientContext());
    assertFalse(SubmarineLogs.isVerbose());

    expectedException.expect(ParseException.class);
    expectedException.expectMessage("cannot be defined for MXNet jobs");
    runJobCli.run(
        new String[]{"--framework", "mxnet",
            "--name", "my-job", "--docker_image", "mxnet-docker:1.1.0",
            "--input_path", "hdfs://input",
            "--num_workers", "2",
            "--worker_launch_cmd", "python run-job.py",
            "--worker_resources", "memory=2048M,vcores=2",
            "--tensorboard_resources", "memory=1024M,vcores=2"});
  }

  @Test
  public void testTensorboardDockerImageCannotBeDefined() throws Exception {
    RunJobCli runJobCli = new RunJobCli(RunJobCliParsingCommonTest.getMockClientContext());
    assertFalse(SubmarineLogs.isVerbose());

    expectedException.expect(ParseException.class);
    expectedException.expectMessage("cannot be defined for MXNet jobs");
    runJobCli.run(
        new String[]{"--framework", "mxnet",
            "--name", "my-job", "--docker_image", "mxnet-docker:1.1.0",
            "--input_path", "hdfs://input",
            "--num_workers", "2",
            "--worker_launch_cmd", "python run-job.py",
            "--worker_resources", "memory=2048M,vcores=2",
            "--tensorboard_docker_image", "TBDockerImage"});
  }
}

