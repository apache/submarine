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
