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

import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.k8s.agent.util.RestClient;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.reflect.TypeToken;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.util.Watch;
import io.kubernetes.client.util.Watch.Response;
import io.kubernetes.client.util.Watchable;
import okhttp3.Call;

public class NotebookHandler extends CustomResourceHandler {

  private static final Logger LOG = LoggerFactory.getLogger(NotebookHandler.class);
  private Watchable<CoreV1Event> watcher;
  private String podName;
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

    try {
       String podLabelSelector = String.format("%s=%s", NotebookCR.NOTEBOOK_ID,
                this.resourceId); 
       V1PodList podList = this.coreV1Api.listNamespacedPod(namespace, null, null, null, null,
               podLabelSelector, null, null, null, null, null);
       this.podName = podList.getItems().get(0).getMetadata().getName();
       String fieldSelector = String.format("involvedObject.name=%s", this.podName);

       Call call =  coreV1Api.listNamespacedEventCall(namespace, null, null, null, fieldSelector,
               null, null, null, null, null, true, null);
       
       watcher = Watch.createWatch(client, call, new TypeToken<Response<CoreV1Event>>(){}.getType());

    } catch (ApiException e) {
      e.printStackTrace();
    }
    restClient = new RestClient(serverHost, serverPort);
  }

  @Override
  public void run() {
    while (true) {    
      for (Response<CoreV1Event> event: watcher) {
          String reason = event.object.getReason();
          switch (reason) {
            case "Created":
            case "Scheduled":    
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, Notebook.Status.STATUS_CREATING.getValue());
              break;
            case "Started":
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, Notebook.Status.STATUS_RUNNING.getValue());
              break;
            case "Failed":
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, Notebook.Status.STATUS_FAILED.getValue());
              break;
            case "Pulling":
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, Notebook.Status.STATUS_PULLING.getValue());
              break;
            case "Killing":
              restClient.callStatusUpdate(CustomResourceType.Notebook, this.resourceId, Notebook.Status.STATUS_TERMINATING.getValue());
              LOG.info("Receive terminating event, exit progress");
              return;
            default:
              LOG.info(String.format("Unprocessed event type:%s", reason));  
          }
      }
    }
  }
}
