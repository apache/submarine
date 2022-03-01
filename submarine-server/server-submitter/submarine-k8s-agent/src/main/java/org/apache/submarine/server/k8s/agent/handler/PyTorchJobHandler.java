package org.apache.submarine.server.k8s.agent.handler;

import java.io.IOException;
import java.util.List;

import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.k8s.agent.util.RestClient;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobList;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.openapi.models.V1JobCondition;
import io.kubernetes.client.util.Watch.Response;
import io.kubernetes.client.util.Watch;
import io.kubernetes.client.util.Watchable;
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import okhttp3.Call;

public class PyTorchJobHandler extends CustomResourceHandler {
  private static final Logger LOG = LoggerFactory.getLogger(PyTorchJobHandler.class);
  private GenericKubernetesApi<PyTorchJob, PyTorchJobList> pytorchJobClient;
  private Watchable<CoreV1Event> watcher;
  public PyTorchJobHandler() throws IOException {
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
    pytorchJobClient =
            new GenericKubernetesApi<>(
                    PyTorchJob.class, PyTorchJobList.class,
                    PyTorchJob.CRD_PYTORCH_GROUP_V1, PyTorchJob.CRD_PYTORCH_VERSION_V1,
                    PyTorchJob.CRD_PYTORCH_PLURAL_V1, client);
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
    Gson gson = new Gson();
    while (true) {
      for (Response<CoreV1Event> event: watcher) {
        PyTorchJob job = pytorchJobClient.get(this.namespace, this.resourceId).getObject();  
        List<V1JobCondition> conditionList = job.getStatus().getConditions();
        V1JobCondition lastCondition = conditionList.get(conditionList.size() - 1);
        Experiment experiment = MLJobConverter.toJobFromMLJob(job);
           
        this.restClient.callStatusUpdate(CustomResourceType.PyTorchJob, resourceId, experiment);
        LOG.info(String.format("receiving condition:%s", lastCondition.getReason()));
        LOG.info(String.format("current status of PyTorchjob:%s is %s", resourceId, experiment.getStatus()));
        
        switch (lastCondition.getReason()) {
          case "PyTorchJobSucceeded":
            LOG.info(String.format("PyTorchjob:%s is succeeded, exit", this.resourceId));
            return;
          case "PyTorchJobFailed":
            LOG.info(String.format("PyTorchjob:%s is failed, exit", this.resourceId));
            return;
          default:    
            break;    
        }
        
      }   
    }
  }
}
