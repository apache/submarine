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
import java.util.List;

import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.k8s.agent.util.RestClient;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobList;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.reflect.TypeToken;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.openapi.models.V1JobCondition;
import io.kubernetes.client.util.Watch.Response;
import io.kubernetes.client.util.Watch;
import io.kubernetes.client.util.Watchable;
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import okhttp3.Call;

public class TFJobHandler extends CustomResourceHandler {
  private static final Logger LOG = LoggerFactory.getLogger(TFJobHandler.class);
  private GenericKubernetesApi<TFJob, TFJobList> tfJobClient;
  private Watchable<CoreV1Event> watcher;
  public TFJobHandler() throws IOException {
    super();
  }


  @Override
  public void init(String serverHost, Integer serverPort,
          String namespace, String crName, String resourceId) {
    this.serverHost = serverHost;
    this.serverPort = serverPort;
    this.namespace = namespace;
    this.crName = crName;
    this.resourceId = resourceId;
    tfJobClient =
            new GenericKubernetesApi<>(
                    TFJob.class, TFJobList.class,
                    TFJob.CRD_TF_GROUP_V1, TFJob.CRD_TF_VERSION_V1,
                    TFJob.CRD_TF_PLURAL_V1, client);
    try {
      String fieldSelector = String.format("involvedObject.name=%s", resourceId);
      LOG.info("fieldSelector:" + fieldSelector);
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
        TFJob job = tfJobClient.get(this.namespace, this.resourceId).getObject();  
        List<V1JobCondition> conditionList = job.getStatus().getConditions();
        V1JobCondition lastCondition = conditionList.get(conditionList.size() - 1);
        Experiment experiment = MLJobConverter.toJobFromMLJob(job);
        
        this.restClient.callStatusUpdate(CustomResourceType.TFJob, resourceId, experiment);
        LOG.info(String.format("receiving condition:%s", lastCondition.getReason()));
        LOG.info(String.format("current status of tfjob:%s is %s", resourceId, experiment.getStatus()));
        
        switch (lastCondition.getReason()) {
          case "TFJobSucceeded":
            LOG.info(String.format("TfJob:%s is succeeded, exit", this.resourceId));
            return;
          case "TFJobFailed":
            LOG.info(String.format("TfJob:%s is failed, exit", this.resourceId));
            return;
          default:    
            break;    
        }
        
      }   
    }
  }
}
