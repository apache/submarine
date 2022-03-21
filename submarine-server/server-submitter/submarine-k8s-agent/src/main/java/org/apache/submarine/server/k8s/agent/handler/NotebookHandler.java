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

package org.apache.submarine.server.k8s.agent.handler;

import java.io.IOException;

import io.kubernetes.client.openapi.models.CoreV1EventList;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.util.generic.options.ListOptions;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.k8s.agent.util.RestClient;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCRList;
import org.apache.submarine.server.submitter.k8s.util.NotebookUtils;
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.util.Watch.Response;
import io.kubernetes.client.util.Watchable;

public class NotebookHandler extends CustomResourceHandler {

  private static final Logger LOG = LoggerFactory.getLogger(NotebookHandler.class);
  private Watchable<CoreV1Event> watcher;

  private GenericKubernetesApi<V1Pod, V1PodList> podClient;
  private GenericKubernetesApi<CoreV1Event, CoreV1EventList> eventClient;
  private GenericKubernetesApi<NotebookCR, NotebookCRList> notebookCRClient;

  private String uid;

  public NotebookHandler() throws IOException {
    super();
  }

  @Override
  public void init(String serverHost, Integer serverPort, String namespace,
          String crName, String resourceId) {
    this.serverHost = serverHost;
    this.serverPort = serverPort;
    this.namespace = namespace;
    this.crName = crName;
    this.resourceId = resourceId;

    podClient =
            new GenericKubernetesApi<>(
                    V1Pod.class, V1PodList.class,
                    "", "v1", "pods", client);
    eventClient =
            new GenericKubernetesApi<>(
                    CoreV1Event.class, CoreV1EventList.class,
                    "", "v1", "events", client);
    notebookCRClient =
            new GenericKubernetesApi<>(
                    NotebookCR.class, NotebookCRList.class,
                    NotebookCR.CRD_NOTEBOOK_GROUP_V1, NotebookCR.CRD_NOTEBOOK_VERSION_V1,
                    NotebookCR.CRD_NOTEBOOK_PLURAL_V1, client);

    try {
      ListOptions listOptions = new ListOptions();
      String podLabelSelector = String.format("%s=%s", NotebookCR.NOTEBOOK_ID, this.resourceId);
      listOptions.setLabelSelector(podLabelSelector);
      
      
      V1PodList podList = podClient.list(namespace, listOptions).throwsApiException().getObject();
      this.uid = podList.getItems().get(podList.getItems().size() - 1).getMetadata().getUid();
      
      listOptions = new ListOptions();
      String fieldSelector = String.format("involvedObject.uid=%s", this.uid);

      listOptions.setFieldSelector(fieldSelector);
      watcher = eventClient.watch(namespace, listOptions);

    } catch (ApiException e) {
      e.printStackTrace();
    }
    restClient = new RestClient(serverHost, serverPort);
  }

  @Override
  public void run() {
    Notebook notebook = null;
    while (true) {
      for (Response<CoreV1Event> event: watcher) {
        String reason = event.object.getReason();
      
        Object object = null;
        try {
          switch (reason) {
            case "Created":
            case "Scheduled":
              object = notebookCRClient.get(namespace, crName).throwsApiException().getObject();
              notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
              notebook.setStatus(Notebook.Status.STATUS_CREATING.getValue());
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, notebook);
              break;
            case "Started":
              object = notebookCRClient.get(namespace, crName).throwsApiException().getObject();
              notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
              notebook.setStatus(Notebook.Status.STATUS_RUNNING.getValue());
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, notebook);
              break;
            case "Failed":
              object = notebookCRClient.get(namespace, crName).throwsApiException().getObject();
              notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
              notebook.setStatus(Notebook.Status.STATUS_FAILED.getValue());
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, notebook);
              break;
            case "Pulling":
              object = notebookCRClient.get(namespace, crName).throwsApiException().getObject();
              notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
              notebook.setStatus(Notebook.Status.STATUS_PULLING.getValue());
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, notebook);
              break;
            case "Killing":
              object = notebookCRClient.get(namespace, crName).throwsApiException().getObject();
              notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
              notebook.setStatus(Notebook.Status.STATUS_TERMINATING.getValue());
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, notebook);
              LOG.info("Receive terminating event, exit progress");
              return;
            default:
              LOG.info(String.format("Unprocessed event type:%s", reason));
          }
        } catch (ApiException e) {
          LOG.error("error while accessing k8s", e);
        }
      }
    }
  }
}
