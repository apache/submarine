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

package org.apache.submarine.server.k8s.agent;

import io.fabric8.kubernetes.api.model.apiextensions.v1.CustomResourceDefinition;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.server.mock.KubernetesServer;
import io.fabric8.kubernetes.internal.KubernetesDeserializer;
import org.apache.submarine.server.k8s.agent.model.notebook.NotebookResource;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SubmitSubmarineNotebookAgentTest {

  private static final Logger LOGGER = LoggerFactory.getLogger(SubmitSubmarineNotebookAgentTest.class);

  @Rule
  public KubernetesServer server = new KubernetesServer(true, true);

  KubernetesClient client;

  @Test
  public void testNotebookAgent() {
    // get client
    client = server.getClient();
    // create k8s client
    KubernetesDeserializer.registerCustomKind("apiextensions.k8s.io/v1beta1", "Notebook", NotebookResource.class);
    CustomResourceDefinition notebookCrd = client
            .apiextensions().v1()
            .customResourceDefinitions()
            .load(getClass().getResourceAsStream("/notebook.yml"))
            .get();
    LOGGER.info("Create Notebook CRD ...");
    client.apiextensions().v1().customResourceDefinitions().create(notebookCrd);

    // TODO(cdmikechen) add notebook reconciler to listen notebook CR
  }

}
