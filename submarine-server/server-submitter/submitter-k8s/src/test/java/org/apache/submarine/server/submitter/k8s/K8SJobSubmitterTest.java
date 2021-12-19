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

import io.kubernetes.client.openapi.ApiException;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.TensorboardInfo;
import org.apache.submarine.server.api.experiment.MlflowInfo;
import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * We have two ways to test submitter for K8s cluster, local and travis CI.
 * <p>
 * For running the tests locally, ensure that: 1. There's a K8s cluster running
 * somewhere 2. Had set the env KUBECONFIG variable 3. The CRDs was created in
 * default namespace. The operator doesn't needs to be running.
 * <p>
 * Use "kubectl -n submarine get tfjob" or "kubectl -n submarine get pytorchjob"
 * to check the status if you comment the deletion job code in method "after()"
 * <p>
 * <p>
 * For the travis CI, we use the kind to setup K8s, more info see '.travis.yml'
 * file. Local: docker run -it --privileged -p 8443:8443 -p 10080:10080
 * bsycorp/kind:latest-1.15 Travis: See '.travis.yml'
 */
public class K8SJobSubmitterTest extends SpecBuilder {
  private static final Logger LOG = LoggerFactory.getLogger(K8SJobSubmitterTest.class);
  private K8sSubmitter submitter;

  @Before
  public void before() throws ApiException {
    submitter = new K8sSubmitter();
    submitter.initialize(null);
  }

  @Ignore // TODO(KUAN-HSUN LI): make sure saving a model before create serve
  @Test
  public void testCreateTensorFlowServe() {
    ServeSpec spec = new ServeSpec();
    spec.setModelName("simple");
    spec.setModelVersion(1);
    spec.setModelType("tensorflow");
    spec.setModelURI("s3://submarine/simple");
    submitter.createServe(spec);
  }

  @Ignore // TODO(KUAN-HSUN LI): make sure saving a model before create serve
  @Test
  public void testDeleteTensorFlowServe() {
    ServeSpec spec = new ServeSpec();
    spec.setModelName("simple");
    spec.setModelVersion(1);
    spec.setModelType("tensorflow");
    submitter.deleteServe(spec);
  }

  @Test
  public void testRunPyTorchJobPerRequest() throws URISyntaxException, IOException,
      SubmarineRuntimeException {
    ExperimentSpec spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, pytorchJobReqFile);
    run(spec);
  }

  @Test
  public void testRunTFJobPerRequest() throws URISyntaxException, IOException, SubmarineRuntimeException {
    ExperimentSpec spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    run(spec);
  }

  @Test
  public void testCreateTFJob() throws IOException, URISyntaxException {
    ExperimentSpec spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfTfboardJobReqFile);
    Experiment experiment = submitter.createExperiment(spec);
  }

  @Test
  public void testDeleteTFJob() throws IOException, URISyntaxException {
    ExperimentSpec spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfTfboardJobReqFile);
    Experiment experiment = submitter.deleteExperiment(spec);
  }

  @Test
  public void testGetTensorboardInfo() throws IOException, URISyntaxException {
    TensorboardInfo tensorboardInfo = submitter.getTensorboardInfo();
  }

  @Test
  public void testGetMlflowInfo() throws IOException, URISyntaxException {
    MlflowInfo mlflowInfo = submitter.getMlflowInfo();
  }

  private void run(ExperimentSpec spec) throws SubmarineRuntimeException {
    // create
    Experiment experimentCreated = submitter.createExperiment(spec);
    Assert.assertNotNull(experimentCreated);

    // find
    Experiment experimentFound = submitter.findExperiment(spec);
    Assert.assertNotNull(experimentFound);
    Assert.assertEquals(experimentCreated.getUid(), experimentFound.getUid());
    Assert.assertEquals(experimentCreated.getAcceptedTime(), experimentFound.getAcceptedTime());

    // delete
    Experiment experimentDeleted = submitter.deleteExperiment(spec);
    Assert.assertNotNull(experimentDeleted);
    Assert.assertEquals(Experiment.Status.STATUS_DELETED.getValue(), experimentDeleted.getStatus());
    Assert.assertEquals(experimentFound.getUid(), experimentDeleted.getUid());
  }
}
