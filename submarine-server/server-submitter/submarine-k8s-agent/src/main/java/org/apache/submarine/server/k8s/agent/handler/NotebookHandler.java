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

import org.apache.submarine.server.k8s.agent.util.RestClient;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.NotebookCRList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.util.Watchable;
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import io.kubernetes.client.util.generic.options.ListOptions;

public class NotebookHandler extends CustomResourceHandler {

  private static final Logger LOG = LoggerFactory.getLogger(NotebookHandler.class);
  private GenericKubernetesApi<NotebookCR, NotebookCRList> notebookClient;
  private Watchable<NotebookCR> watcher;
  
  public NotebookHandler() throws IOException {
    super();
    notebookClient = new GenericKubernetesApi<>(NotebookCR.class, NotebookCRList.class
            , NotebookCR.CRD_NOTEBOOK_GROUP_V1, NotebookCR.CRD_APIVERSION_V1
            , NotebookCR.CRD_NOTEBOOK_PLURAL_V1, this.client);
    
  }

  @Override
  public void init(String serverHost, Integer serverPort, String namespace, String crName) {
     
    ListOptions listOption = new ListOptions();
    listOption.setFieldSelector(String.format("metadata.name=%s", crName));
    try {
      watcher = notebookClient.watch(namespace, listOption);
    } catch (ApiException e) {
        e.printStackTrace();
    }
    restClient = new RestClient(serverHost, serverPort);
    
  }

  @Override
  public void run() {
      int i = 0;
      ObjectMapper mapper = new ObjectMapper();
      
      while(watcher.hasNext()) {
          try {
            LOG.info(mapper.writeValueAsString(watcher.next()));
          } catch (JsonProcessingException e ) {
            e.printStackTrace();
          }
          i ++;
          if (i >= 5) {
              break;
          }
      }
  }

  @Override
  public void onAddEvent() {
    
  }

  @Override
  public void onModifyEvent() {
  
      
  }

  @Override
  public void onDeleteEvent() {
 
  }
}
