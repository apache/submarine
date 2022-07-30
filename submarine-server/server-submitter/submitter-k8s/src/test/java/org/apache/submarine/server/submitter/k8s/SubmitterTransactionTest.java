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

import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.model.K8sResource;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SubmitterTransactionTest {

  private static final Logger LOG = LoggerFactory.getLogger(SubmitterTransactionTest.class);

  private K8sSubmitter submitter;

  @Before
  public void before() {
    try {
      submitter = new K8sSubmitter();
      submitter.initialize(null);
    } catch (Exception e) {
      LOG.warn("Init K8sSubmitter failed, but we can continue", e);
    }
  }

  @Test
  public void testOneSuccessCommit() {
    submitter.resourceTransaction(new SuccessResource("test1"));
  }

  @Test
  public void testCombineCommit() {
    SuccessResource resource1 = new SuccessResource("test1");
    FailedResource resource2 = new FailedResource();
    try {
      submitter.resourceTransaction(resource1, resource2);
    } catch (Exception e) {
      e.printStackTrace();
    }
    Assert.assertFalse(resource1.isCommit());
  }

  @Test
  public void testCommitOrder() {
    SuccessResource resource1 = new SuccessResource("test1");
    FailedResource resource2 = new FailedResource();
    SuccessResource resource3 = new SuccessResource("test3\"");
    try {
      submitter.resourceTransaction(resource1, resource2, resource3);
    } catch (Exception e) {
      e.printStackTrace();
    }
    Assert.assertFalse(resource1.isCommit());
    Assert.assertFalse(resource3.isCommit());
  }

}

class SuccessResource implements K8sResource<SuccessResource> {

  private boolean commit = false;

  public boolean isCommit() {
    return commit;
  }

  private final String name;

  SuccessResource(String name) {
    this.name = name;
  }

  @Override
  public String getKind() {
    return "Success";
  }

  @Override
  public V1ObjectMeta getMetadata() {
    return K8sSubmitter.createMeta("default", name);
  }

  @Override
  public SuccessResource read(K8sClient api) {
    return this;
  }

  @Override
  public SuccessResource create(K8sClient api) {
    this.commit = true;
    return this;
  }

  @Override
  public SuccessResource replace(K8sClient api) {
    this.commit = true;
    return this;
  }

  @Override
  public SuccessResource delete(K8sClient api) {
    this.commit = false;
    return this;
  }
}

class FailedResource implements K8sResource<FailedResource> {

  @Override
  public String getKind() {
    return "Failed";
  }

  @Override
  public V1ObjectMeta getMetadata() {
    return K8sSubmitter.createMeta("default", "test2");
  }

  @Override
  public FailedResource read(K8sClient api) {
    return this;
  }

  @Override
  public FailedResource create(K8sClient api) {
    throw new SubmarineRuntimeException("failed create!");
  }

  @Override
  public FailedResource replace(K8sClient api) {
    return this;
  }

  @Override
  public FailedResource delete(K8sClient api) {
    return this;
  }
}
