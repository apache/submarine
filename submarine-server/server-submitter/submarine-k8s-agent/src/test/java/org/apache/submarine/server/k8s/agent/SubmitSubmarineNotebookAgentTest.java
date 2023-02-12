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

import io.fabric8.kubernetes.api.model.ObjectMeta;
import io.fabric8.kubernetes.api.model.ObjectMetaBuilder;
import io.fabric8.kubernetes.api.model.OwnerReferenceBuilder;
import io.fabric8.kubernetes.api.model.apiextensions.v1.CustomResourceDefinition;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.server.mock.KubernetesServer;
import io.fabric8.kubernetes.internal.KubernetesDeserializer;
import io.javaoperatorsdk.operator.Operator;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.k8s.agent.model.notebook.NotebookResource;
import org.apache.submarine.server.k8s.agent.model.notebook.status.NotebookCondition;
import org.apache.submarine.server.k8s.agent.model.notebook.status.NotebookStatus;
import org.apache.submarine.server.k8s.agent.reconciler.NotebookReconciler;
import org.apache.submarine.server.k8s.utils.OwnerReferenceConfig;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.takes.facets.fork.FkRegex;
import org.takes.facets.fork.TkFork;
import org.takes.http.Exit;
import org.takes.http.FtBasic;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;

public class SubmitSubmarineNotebookAgentTest {

  private static final Logger LOGGER = LoggerFactory.getLogger(SubmitSubmarineNotebookAgentTest.class);

  @Rule
  public KubernetesServer server = new KubernetesServer(true, true);

  KubernetesClient client;

  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  protected final String H2_JDBC_URL = "jdbc:h2:mem:submarine-test;MODE=MYSQL;DB_CLOSE_DELAY=-1";
  protected final String H2_JDBC_DRIVERCLASS = "org.h2.Driver";
  protected final String H2_JDBC_USERNAME = "root";
  protected final String H2_JDBC_PASSWORD = "";

  @Before
  public void beforeInit() {
    conf.setJdbcUrl(H2_JDBC_URL);
    conf.setJdbcDriverClassName(H2_JDBC_DRIVERCLASS);
    conf.setJdbcUserName(H2_JDBC_USERNAME);
    conf.setJdbcPassword(H2_JDBC_PASSWORD);
    try (Connection conn = DriverManager.getConnection(H2_JDBC_URL,
            H2_JDBC_USERNAME, H2_JDBC_PASSWORD);
         Statement stmt = conn.createStatement()) {
      stmt.execute("RUNSCRIPT FROM 'classpath:/db/notebook.sql'");
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  @Test
  public void testNotebookAgent() throws IOException {
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

    Operator operator = new Operator(client);
    // add notebook reconciler to listen notebook CR
    operator.register(new NotebookReconciler());
    // start operator
    operator.start();

    // add notebook
    NotebookStatus status = new NotebookStatus();
    status.setReadyReplicas(1);
    NotebookCondition condition = new NotebookCondition();
    condition.setType("Running");
    condition.setLastProbeTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    status.setConditions(List.of(condition));
    ObjectMeta meta = new ObjectMetaBuilder()
            .withName("test-notebook")
            .withNamespace(client.getNamespace())
            .withLabels(Map.of("notebook-id", "0490d994-3da0-4fe8-b952-0e4a77268429",
                    "notebook-owner-id", "d9efda1f-c965-476a-a214-3c06570e4261"))
            .addToOwnerReferences(new OwnerReferenceBuilder()
                    .withUid(OwnerReferenceConfig.getSubmarineUid())
                    .withApiVersion(OwnerReferenceConfig.DEFAULT_SUBMARINE_APIVERSION)
                    .withKind(OwnerReferenceConfig.DEFAULT_SUBMARINE_KIND)
                    .build())
            .build();
    NotebookResource resource = new NotebookResource();
    resource.setMetadata(meta);
    resource.setStatus(status);
    client.resource(resource).createOrReplace();

    // after left 10s
    new FtBasic(
            new TkFork(new FkRegex("/health", "ALL GOOD.")), 8080
    ).start(new Exit() {
      private final long max = 10 * 1000;
      private final long start = System.currentTimeMillis();

      @Override
      public boolean ready() {
        return System.currentTimeMillis() - this.start > this.max;
      }
    });
  }

}
