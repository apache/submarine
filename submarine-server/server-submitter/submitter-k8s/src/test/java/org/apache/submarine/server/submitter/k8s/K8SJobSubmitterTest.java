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

import org.apache.submarine.server.submitter.k8s.model.CustomResourceJob;
import org.apache.submarine.server.submitter.k8s.model.CustomResourceJobList;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;

public class K8SJobSubmitterTest {
  private final String jobName = "mnist";
  private K8sJobSubmitter submitter;
  private K8sJobRequest.Path path;

  /**
   * We have two ways to test submitter for K8s cluster, local and travis CI.
   * For the travis CI, we use the kind to setup K8s, more info see '.travis.yml' file.
   *  Local: docker run -it --privileged -p 8443:8443 -p 10080:10080 bsycorp/kind:latest-1.15
   *  Travis: See '.travis.yml'
   * @throws IOException IO
   */
  @Before
  public void before() throws IOException {
    String confPath = System.getProperty("user.home") + "/.kube/config";
    if (!new File(confPath).exists()) {
      throw new IOException("Get kube config file failed.");
    }

    submitter = new K8sJobSubmitter(confPath);
    submitter.initialize(null);
    path = new K8sJobRequest.Path("kubeflow.org", "v1", "kubeflow", "tfjobs");
  }

  @Test
  public void testCreateCustomJob() throws URISyntaxException {
    if (getCustomJob() != null) {
      K8sJobRequest request = new K8sJobRequest(path, null, jobName);
      CustomResourceJob delJob = submitter.deleteCustomResourceJob(request);
      Assert.assertNotNull(delJob);
    }

    CustomResourceJob job = submitter.createCustomJob(new K8sJobRequest(path, getCustomJobSpecFile()));
    Assert.assertNotNull(job);
  }

  @Test
  public void testGetCustomJob() throws URISyntaxException {
    testCreateCustomJob();

    CustomResourceJob job = getCustomJob();
    Assert.assertNotNull(job);
    Assert.assertEquals(job.getMetadata().getName(), jobName);
  }

  @Test
  public void testListCustomJobs() throws URISyntaxException {
    CustomResourceJobList list
        = submitter.listCustomResourceJobs(new K8sJobRequest(path, getCustomJobSpecFile()));
    Assert.assertNotNull(list);
  }

  @Test
  public void testDeleteCustomJob() throws URISyntaxException {
    if (getCustomJob() == null) {
      CustomResourceJob job = submitter.createCustomJob(new K8sJobRequest(path, getCustomJobSpecFile()));
      Assert.assertNotNull(job);
    }

    K8sJobRequest request = new K8sJobRequest(path, null, jobName);
    CustomResourceJob delJob = submitter.deleteCustomResourceJob(request);
    Assert.assertNotNull(delJob);
  }

  private CustomResourceJob getCustomJob() {
    K8sJobRequest request = new K8sJobRequest(path, null, jobName);
    return submitter.getCustomResourceJob(request);
  }

  private File getCustomJobSpecFile() throws URISyntaxException {
    URL fileUrl = this.getClass().getResource("/tf_job_mnist.json");
    return new File(fileUrl.toURI());
  }
}
