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
import io.javaoperatorsdk.operator.Operator;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.database.experiment.entity.ExperimentEntity;
import org.apache.submarine.server.database.experiment.mappers.ExperimentMapper;
import org.apache.submarine.server.database.notebook.entity.NotebookEntity;
import org.apache.submarine.server.database.notebook.mappers.NotebookMapper;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.k8s.agent.model.notebook.NotebookResource;
import org.apache.submarine.server.k8s.agent.model.notebook.status.NotebookCondition;
import org.apache.submarine.server.k8s.agent.model.notebook.status.NotebookStatus;
import org.apache.submarine.server.k8s.agent.model.training.resource.PyTorchJob;
import org.apache.submarine.server.k8s.agent.model.training.resource.TFJob;
import org.apache.submarine.server.k8s.agent.model.training.resource.XGBoostJob;
import org.apache.submarine.server.k8s.agent.model.training.status.JobCondition;
import org.apache.submarine.server.k8s.agent.model.training.status.JobStatus;
import org.apache.submarine.server.k8s.agent.model.training.status.ReplicaStatus;
import org.apache.submarine.server.k8s.agent.reconciler.NotebookReconciler;
import org.apache.submarine.server.k8s.agent.reconciler.PyTorchJobReconciler;
import org.apache.submarine.server.k8s.agent.reconciler.TFJobReconciler;
import org.apache.submarine.server.k8s.agent.reconciler.XGBoostJobReconciler;
import org.apache.submarine.server.k8s.utils.OwnerReferenceConfig;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class SubmitSubmarineAgentTest {

  private static final Logger LOGGER = LoggerFactory.getLogger(SubmitSubmarineAgentTest.class);

  @ClassRule
  public static KubernetesServer server = new KubernetesServer(true, true);

  private static KubernetesClient client;

  private static Operator operator;

  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  private static final String H2_JDBC_URL = "jdbc:h2:mem:submarine-test;MODE=MYSQL;DB_CLOSE_DELAY=-1";
  private static final String H2_JDBC_DRIVERCLASS = "org.h2.Driver";
  private static final String H2_JDBC_USERNAME = "root";
  private static final String H2_JDBC_PASSWORD = "";

  @BeforeClass
  public static void beforeInit() {
    // setup h2 database
    conf.setJdbcUrl(H2_JDBC_URL);
    conf.setJdbcDriverClassName(H2_JDBC_DRIVERCLASS);
    conf.setJdbcUserName(H2_JDBC_USERNAME);
    conf.setJdbcPassword(H2_JDBC_PASSWORD);
    try (Connection conn = DriverManager.getConnection(H2_JDBC_URL,
            H2_JDBC_USERNAME, H2_JDBC_PASSWORD);
         Statement stmt = conn.createStatement()) {
      stmt.execute("RUNSCRIPT FROM 'classpath:/db/agent-init.sql'");
    } catch (SQLException e) {
      e.printStackTrace();
    }

    // set client and operator
    client = server.getClient();
    operator = new Operator(client, null);

    // create notbook resource
    client.getKubernetesSerialization().registerKubernetesResource("apiextensions.k8s.io/v1","Notebook", NotebookResource.class);
    CustomResourceDefinition notebookCrd = client
            .apiextensions().v1()
            .customResourceDefinitions()
            .load(SubmitSubmarineAgentTest.class.getResourceAsStream("/custom-resources/notebook.yml"))
            .item();
    LOGGER.info("Create Notebook CRD ...");
    client.apiextensions().v1().customResourceDefinitions().createOrReplace(notebookCrd);

    // create tf resource
    client.getKubernetesSerialization().registerKubernetesResource("apiextensions.k8s.io/v1", "TFJob", TFJob.class);
    CustomResourceDefinition tfCrd = client
            .apiextensions().v1()
            .customResourceDefinitions()
            .load(SubmitSubmarineAgentTest.class.getResourceAsStream("/custom-resources/tfjobs.yaml"))
            .item();
    LOGGER.info("Create TF CRD ...");
    client.apiextensions().v1().customResourceDefinitions().create(tfCrd);

    // create pytorch resource
    client.getKubernetesSerialization().registerKubernetesResource("apiextensions.k8s.io/v1", "PyTorchJob", PyTorchJob.class);
    CustomResourceDefinition ptCrd = client
            .apiextensions().v1()
            .customResourceDefinitions()
            .load(SubmitSubmarineAgentTest.class.getResourceAsStream("/custom-resources/pytorchjobs.yaml"))
            .item();
    LOGGER.info("Create PyTorch CRD ...");
    client.apiextensions().v1().customResourceDefinitions().create(ptCrd);

    // create xgboost resource
    client.getKubernetesSerialization().registerKubernetesResource("apiextensions.k8s.io/v1", "XGBoostJob", XGBoostJob.class);
    CustomResourceDefinition xgbCrd = client
            .apiextensions().v1()
            .customResourceDefinitions()
            .load(SubmitSubmarineAgentTest.class.getResourceAsStream("/custom-resources/xgboostjobs.yaml"))
            .item();
    LOGGER.info("Create XGBoost CRD ...");
    client.apiextensions().v1().customResourceDefinitions().create(xgbCrd);

    // add reconcilers to listen custom resources
    operator.register(new NotebookReconciler());
    operator.register(new TFJobReconciler());
    operator.register(new PyTorchJobReconciler());
    operator.register(new XGBoostJobReconciler());

    // start operator
    operator.start();
  }

  @Test
  public void testTfJobAgent() throws InterruptedException {
    // add notebook
    JobStatus status = new JobStatus();
    JobCondition condition = new JobCondition();
    condition.setMessage("TFJob test/experiment-1659167632755-0001 is running.");
    condition.setReason("TFJobRunning");
    condition.setStatus("True");
    condition.setType("Running");
    condition.setLastTransitionTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    condition.setLastUpdateTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    status.setConditions(List.of(condition));
    status.setReplicaStatuses(Map.of("PS", new ReplicaStatus(1, 0, 0),
            "Worker", new ReplicaStatus(1, 0, 0)));
    ObjectMeta meta = new ObjectMetaBuilder()
            .withName("experiment-1659167632755-0001")
            .withNamespace(client.getNamespace())
            .withLabels(Map.of("submarine-experiment-name", "test-tfjob"))
            .addToOwnerReferences(new OwnerReferenceBuilder()
                    .withUid(OwnerReferenceConfig.getSubmarineUid())
                    .withApiVersion(OwnerReferenceConfig.DEFAULT_SUBMARINE_APIVERSION)
                    .withKind(OwnerReferenceConfig.DEFAULT_SUBMARINE_KIND)
                    .build())
            .build();
    TFJob resource = new TFJob();
    resource.setMetadata(meta);
    resource.setStatus(status);
    client.resource(resource).create();
    client.resource(resource).updateStatus();

    // left 5s to process
    Thread.sleep(TimeUnit.SECONDS.toMillis(5));

    // check status have changed
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      ExperimentEntity tfjob = mapper.select("experiment-1659167632755-0001");
      Assert.assertEquals("Running", tfjob.getExperimentStatus());
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
      throw e;
    }
  }

  @Test
  public void testPytorchJobAgent() throws InterruptedException {
    // add notebook
    JobStatus status = new JobStatus();
    JobCondition condition = new JobCondition();
    condition.setMessage("PytorchJob test/experiment-1659167632755-0002 is running.");
    condition.setReason("PytorchJobRunning");
    condition.setStatus("True");
    condition.setType("Running");
    condition.setLastTransitionTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    condition.setLastUpdateTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    status.setConditions(List.of(condition));
    ObjectMeta meta = new ObjectMetaBuilder()
            .withName("experiment-1659167632755-0002")
            .withNamespace(client.getNamespace())
            .withLabels(Map.of("submarine-experiment-name", "test-pytorchjob"))
            .addToOwnerReferences(new OwnerReferenceBuilder()
                    .withUid(OwnerReferenceConfig.getSubmarineUid())
                    .withApiVersion(OwnerReferenceConfig.DEFAULT_SUBMARINE_APIVERSION)
                    .withKind(OwnerReferenceConfig.DEFAULT_SUBMARINE_KIND)
                    .build())
            .build();
    PyTorchJob resource = new PyTorchJob();
    resource.setMetadata(meta);
    resource.setStatus(status);
    client.resource(resource).create();
    client.resource(resource).updateStatus();

    // left 5s to process
    Thread.sleep(TimeUnit.SECONDS.toMillis(5));

    // check status have changed
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      ExperimentEntity tfjob = mapper.select("experiment-1659167632755-0002");
      Assert.assertEquals("Running", tfjob.getExperimentStatus());
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
      throw e;
    }
  }

  @Test
  public void testXGBoostJobAgent() throws InterruptedException {
    // add notebook
    JobStatus status = new JobStatus();
    JobCondition condition = new JobCondition();
    condition.setMessage("XGBoostJob test/experiment-1659167632755-0003 is running.");
    condition.setReason("XGBoostJobRunning");
    condition.setStatus("True");
    condition.setType("Running");
    condition.setLastTransitionTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    condition.setLastUpdateTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    status.setConditions(List.of(condition));
    ObjectMeta meta = new ObjectMetaBuilder()
            .withName("experiment-1659167632755-0003")
            .withNamespace(client.getNamespace())
            .withLabels(Map.of("submarine-experiment-name", "test-xgboostjob"))
            .addToOwnerReferences(new OwnerReferenceBuilder()
                    .withUid(OwnerReferenceConfig.getSubmarineUid())
                    .withApiVersion(OwnerReferenceConfig.DEFAULT_SUBMARINE_APIVERSION)
                    .withKind(OwnerReferenceConfig.DEFAULT_SUBMARINE_KIND)
                    .build())
            .build();
    XGBoostJob resource = new XGBoostJob();
    resource.setMetadata(meta);
    resource.setStatus(status);
    client.resource(resource).create();
    client.resource(resource).updateStatus();

    // left 5s to process
    Thread.sleep(TimeUnit.SECONDS.toMillis(5));

    // check status have changed
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      ExperimentEntity tfjob = mapper.select("experiment-1659167632755-0003");
      Assert.assertEquals("Running", tfjob.getExperimentStatus());
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
      throw e;
    }
  }


  /**
   * This can test notebook-controller 1.4.0
   */
  @Test
  public void testNotebookAgent() throws InterruptedException {
    // add notebook
    NotebookStatus status = new NotebookStatus();
    status.setReadyReplicas(1);
    NotebookCondition condition = new NotebookCondition();
    condition.setType("Running");
    condition.setLastProbeTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    status.setConditions(List.of(condition));
    ObjectMeta meta = new ObjectMetaBuilder()
            .withName("notebook-1642402491519-0001-test-notebook")
            .withNamespace(client.getNamespace())
            .withLabels(Map.of("notebook-id", "notebook_1642402491519_0001",
                    "notebook-owner-id", "e9ca23d68d884d4ebb19d07889727dae"))
            .addToOwnerReferences(new OwnerReferenceBuilder()
                    .withUid(OwnerReferenceConfig.getSubmarineUid())
                    .withApiVersion(OwnerReferenceConfig.DEFAULT_SUBMARINE_APIVERSION)
                    .withKind(OwnerReferenceConfig.DEFAULT_SUBMARINE_KIND)
                    .build())
            .build();
    NotebookResource resource = new NotebookResource();
    resource.setMetadata(meta);
    resource.setStatus(status);
    client.resource(resource).create();
    client.resource(resource).updateStatus();

    // left 5s to process
    Thread.sleep(TimeUnit.SECONDS.toMillis(5));

    // check status have changed
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      NotebookEntity notebook = mapper.select("notebook_1642402491519_0001");
      Assert.assertEquals("running", notebook.getNotebookStatus());
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
      throw e;
    }
  }

  /**
   * This can test notebook-controller 1.7.0
   */
  @Test
  public void testNotebookAgentNewConditions() throws InterruptedException {
    // add notebook
    NotebookStatus status = new NotebookStatus();
    status.setReadyReplicas(1);

    String probeTime = LocalDateTime.now().atZone(ZoneOffset.UTC).toString();
    NotebookCondition condition1 = new NotebookCondition();
    condition1.setType("Initialized");
    condition1.setLastProbeTime(probeTime);
    condition1.setLastTransitionTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    NotebookCondition condition2 = new NotebookCondition();
    condition2.setType("Ready");
    condition2.setLastProbeTime(probeTime);
    condition2.setLastTransitionTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());
    NotebookCondition condition3 = new NotebookCondition();
    condition3.setType("ContainersReady");
    condition3.setLastProbeTime(probeTime);
    condition3.setLastTransitionTime(LocalDateTime.now().atZone(ZoneOffset.UTC).toString());

    status.setConditions(List.of(condition1, condition2, condition3));
    ObjectMeta meta = new ObjectMetaBuilder()
            .withName("notebook-1642402491519-0002-test-notebook")
            .withNamespace(client.getNamespace())
            .withLabels(Map.of("notebook-id", "notebook_1642402491519_0002",
                    "notebook-owner-id", "e9ca23d68d884d4ebb19d07889727dae"))
            .addToOwnerReferences(new OwnerReferenceBuilder()
                    .withUid(OwnerReferenceConfig.getSubmarineUid())
                    .withApiVersion(OwnerReferenceConfig.DEFAULT_SUBMARINE_APIVERSION)
                    .withKind(OwnerReferenceConfig.DEFAULT_SUBMARINE_KIND)
                    .build())
            .build();
    NotebookResource resource = new NotebookResource();
    resource.setMetadata(meta);
    resource.setStatus(status);
    client.resource(resource).create();
    client.resource(resource).updateStatus();

    // left 5s to process
    Thread.sleep(TimeUnit.SECONDS.toMillis(5));

    // check status have changed
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      NotebookEntity notebook = mapper.select("notebook_1642402491519_0002");
      Assert.assertEquals("running", notebook.getNotebookStatus());
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
      throw e;
    }
  }

  @AfterClass
  public static void close() {
    operator.stop();
    client.close();
  }

}
