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

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import io.kubernetes.client.ApiClient;
import io.kubernetes.client.ApiException;
import io.kubernetes.client.Configuration;
import io.kubernetes.client.JSON;
import io.kubernetes.client.apis.AppsV1Api;
import io.kubernetes.client.apis.CoreV1Api;
import io.kubernetes.client.apis.CustomObjectsApi;
import io.kubernetes.client.models.V1DeleteOptionsBuilder;
import io.kubernetes.client.models.V1Deployment;
import io.kubernetes.client.models.V1ObjectMeta;
import io.kubernetes.client.models.V1PersistentVolume;
import io.kubernetes.client.models.V1PersistentVolumeClaim;
import io.kubernetes.client.models.V1PersistentVolumeClaimVolumeSource;
import io.kubernetes.client.models.V1Pod;
import io.kubernetes.client.models.V1PodList;
import io.kubernetes.client.models.V1Service;
import io.kubernetes.client.models.V1Status;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.experiment.TensorboardInfo;
import org.apache.submarine.server.api.experiment.MlflowInfo;
import org.apache.submarine.server.api.experiment.Serve;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRoute;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRouteSpec;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.SpecRoute;
import org.apache.submarine.server.submitter.k8s.model.middlewares.Middlewares;
import org.apache.submarine.server.submitter.k8s.parser.ExperimentSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.NotebookSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.ServeSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.VolumeSpecParser;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.apache.submarine.server.submitter.k8s.util.NotebookUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * JobSubmitter for Kubernetes Cluster.
 */
public class K8sSubmitter implements Submitter {
  private static final Logger LOG = LoggerFactory.getLogger(K8sSubmitter.class);

  private static final String KUBECONFIG_ENV = "KUBECONFIG";

  private static final String TF_JOB_SELECTOR_KEY = "tf-job-name=";
  private static final String PYTORCH_JOB_SELECTOR_KEY = "pytorch-job-name=";
  
  private static final String ENV_NAMESPACE = "ENV_NAMESPACE";

  // K8s API client for CRD
  private CustomObjectsApi api;

  private CoreV1Api coreApi;

  private AppsV1Api appsV1Api;

  public K8sSubmitter() {}

  @Override
  public void initialize(SubmarineConfiguration conf) {
    ApiClient client = null;
    try {
      String path = System.getenv(KUBECONFIG_ENV);
      KubeConfig config = KubeConfig.loadKubeConfig(new FileReader(path));
      client = ClientBuilder.kubeconfig(config).build();
    } catch (Exception e) {
      LOG.info("Maybe in cluster mode, try to initialize the client again.");
      try {
        client = ClientBuilder.cluster().build();
      } catch (IOException e1) {
        LOG.error("Initialize K8s submitter failed. " + e.getMessage(), e1);
        throw new SubmarineRuntimeException(500, "Initialize K8s submitter failed.");
      }
    } finally {
      Configuration.setDefaultApiClient(client);
    }

    if (api == null) {
      api = new CustomObjectsApi();
    }
    if (coreApi == null) {
      coreApi = new CoreV1Api(client);
    }

    if (appsV1Api == null) {
      appsV1Api = new AppsV1Api();
    }

    client.setDebugging(true);
  }

  @Override
  public Experiment createExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    Experiment experiment;
    try {
      MLJob mlJob = ExperimentSpecParser.parseJob(spec);
      Object object = api.createNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob, "true");
      experiment = parseExperimentResponseObject(object, ParseOp.PARSE_OP_RESULT);
    } catch (InvalidSpecException e) {
      LOG.error("K8s submitter: parse Job object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(400, e.getMessage());
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse Job object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse Job object failed by " +
          e.getMessage());
    }
    return experiment;
  }

  @Override
  public Experiment findExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    Experiment experiment;
    try {
      MLJob mlJob = ExperimentSpecParser.parseJob(spec);
      Object object = api.getNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob.getMetadata().getName());
      experiment = parseExperimentResponseObject(object, ParseOp.PARSE_OP_RESULT);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return experiment;
  }

  @Override
  public Experiment patchExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    Experiment experiment;
    try {
      MLJob mlJob = ExperimentSpecParser.parseJob(spec);
      Object object = api.patchNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob.getMetadata().getName(),
          mlJob);
      experiment = parseExperimentResponseObject(object, ParseOp.PARSE_OP_RESULT);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return experiment;
  }

  @Override
  public Experiment deleteExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    Experiment experiment;
    try {
      MLJob mlJob = ExperimentSpecParser.parseJob(spec);
      Object object = api.deleteNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob.getMetadata().getName(),
          MLJobConverter.toDeleteOptionsFromMLJob(mlJob), null, null, null);
      experiment = parseExperimentResponseObject(object, ParseOp.PARSE_OP_DELETE);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return experiment;
  }

  private Experiment parseExperimentResponseObject(Object object, ParseOp op)
          throws SubmarineRuntimeException {
    Gson gson = new JSON().getGson();
    String jsonString = gson.toJson(object);
    LOG.info("Upstream response JSON: {}", jsonString);
    try {
      if (op == ParseOp.PARSE_OP_RESULT) {
        MLJob mlJob = gson.fromJson(jsonString, MLJob.class);
        return MLJobConverter.toJobFromMLJob(mlJob);
      } else if (op == ParseOp.PARSE_OP_DELETE) {
        V1Status status = gson.fromJson(jsonString, V1Status.class);
        return MLJobConverter.toJobFromStatus(status);
      }
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
    }
    throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
  }

  @Override
  public ExperimentLog getExperimentLogName(ExperimentSpec spec, String id) {
    ExperimentLog experimentLog = new ExperimentLog();
    experimentLog.setExperimentId(id);
    try {
      final V1PodList podList = coreApi.listNamespacedPod(
          spec.getMeta().getNamespace(),
          "false", null, null,
          getJobLabelSelector(spec), null, null,
          null, null);
      for (V1Pod pod: podList.getItems()) {
        String podName = pod.getMetadata().getName();
        experimentLog.addPodLog(podName, null);
      }
    } catch (final ApiException e) {
      LOG.error("Error when listing pod for experiment:" + spec.getMeta().getName(), e.getMessage());
    }
    return experimentLog;
  }

  @Override
  public ExperimentLog getExperimentLog(ExperimentSpec spec, String id) {
    ExperimentLog experimentLog = new ExperimentLog();
    experimentLog.setExperimentId(id);
    try {
      final V1PodList podList = coreApi.listNamespacedPod(
          spec.getMeta().getNamespace(),
          "false", null, null,
          getJobLabelSelector(spec), null, null,
          null, null);

      for (V1Pod pod : podList.getItems()) {
        String podName = pod.getMetadata().getName();
        String namespace = pod.getMetadata().getNamespace();
        String podLog = coreApi.readNamespacedPodLog(
            podName, namespace, null, Boolean.FALSE,
            Integer.MAX_VALUE, null, Boolean.FALSE,
            Integer.MAX_VALUE, null, Boolean.FALSE);

        experimentLog.addPodLog(podName, podLog);
      }
    } catch (final ApiException e) {
      LOG.error("Error when listing pod for experiment:" + spec.getMeta().getName(), e.getMessage());
    }
    return experimentLog;
  }

  @Override
  public TensorboardInfo getTensorboardInfo() throws SubmarineRuntimeException {
    final String name = "submarine-tensorboard";
    final String ingressRouteName = "submarine-tensorboard-ingressroute";
    String namespace = "default";
    if (System.getenv(ENV_NAMESPACE) != null) {
      namespace = System.getenv(ENV_NAMESPACE);
    }

    try {
      V1Deployment deploy =  appsV1Api.readNamespacedDeploymentStatus(name, namespace, "true");
      boolean available = deploy.getStatus().getAvailableReplicas() > 0; // at least one replica is running

      IngressRoute ingressRoute = new IngressRoute();
      V1ObjectMeta meta = new V1ObjectMeta();
      meta.setName(ingressRouteName);
      meta.setNamespace(namespace);
      ingressRoute.setMetadata(meta);
      Object object = api.getNamespacedCustomObject(
          ingressRoute.getGroup(), ingressRoute.getVersion(),
          ingressRoute.getMetadata().getNamespace(),
          ingressRoute.getPlural(), ingressRouteName
      );

      Gson gson = new JSON().getGson();
      String jsonString = gson.toJson(object);
      IngressRoute result = gson.fromJson(jsonString, IngressRoute.class);


      String route = result.getSpec().getRoutes().stream().findFirst().get().getMatch();

      //  replace "PathPrefix(`/tensorboard`)" with "/tensorboard/"
      String url = route.replace("PathPrefix(`", "").replace("`)", "/");

      TensorboardInfo tensorboardInfo = new TensorboardInfo(available, url);

      return tensorboardInfo;
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public MlflowInfo getMlflowInfo() throws SubmarineRuntimeException {
    final String name = "submarine-mlflow";
    final String ingressRouteName = "submarine-mlflow-ingressroute";
    String namespace = "default";
    if (System.getenv(ENV_NAMESPACE) != null) {
      namespace = System.getenv(ENV_NAMESPACE);
    }

    try {
      V1Deployment deploy =  appsV1Api.readNamespacedDeploymentStatus(name, namespace, "true");
      boolean available = deploy.getStatus().getAvailableReplicas() > 0; // at least one replica is running

      IngressRoute ingressRoute = new IngressRoute();
      V1ObjectMeta meta = new V1ObjectMeta();
      meta.setName(ingressRouteName);
      meta.setNamespace(namespace);
      ingressRoute.setMetadata(meta);
      Object object = api.getNamespacedCustomObject(
              ingressRoute.getGroup(), ingressRoute.getVersion(),
              ingressRoute.getMetadata().getNamespace(),
              ingressRoute.getPlural(), ingressRouteName
      );

      Gson gson = new JSON().getGson();
      String jsonString = gson.toJson(object);
      IngressRoute result = gson.fromJson(jsonString, IngressRoute.class);


      String route = result.getSpec().getRoutes().stream().findFirst().get().getMatch();

      String url = route.replace("PathPrefix(`", "").replace("`)", "/");

      MlflowInfo mlflowInfo = new MlflowInfo(available, url);

      return mlflowInfo;
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }


  @Override
  public Notebook createNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    Notebook notebook;
    final String name = spec.getMeta().getName();
    final String pvName = NotebookUtils.PV_PREFIX + name;
    final String host = NotebookUtils.HOST_PATH;
    final String storage = NotebookUtils.STORAGE;
    final String pvcName = NotebookUtils.PVC_PREFIX + name;
    String namespace = "default";
    
    if (System.getenv(ENV_NAMESPACE) != null) {
      namespace = System.getenv(ENV_NAMESPACE);
    }
    
    try {
      // create notebook custom resource
      NotebookCR notebookCR = NotebookSpecParser.parseNotebook(spec);
      Map<String, String> labels = new HashMap<>();
      labels.put(NotebookCR.NOTEBOOK_OWNER_SELECTOR_KET, spec.getMeta().getOwnerId());
      notebookCR.getMetadata().setLabels(labels);
      notebookCR.getMetadata().setNamespace(namespace);
      
      // create persistent volume
      createPersistentVolume(pvName, host, storage);

      // create persistent volume claim
      createPersistentVolumeClaim(pvcName, namespace, pvName, storage);

      // bind persistent volume claim
      V1PersistentVolumeClaimVolumeSource pvcSource = new V1PersistentVolumeClaimVolumeSource()
              .claimName(pvcName);
      notebookCR.getSpec().getTemplate().getSpec().getVolumes().get(0).persistentVolumeClaim(pvcSource);

      Object object = api.createNamespacedCustomObject(notebookCR.getGroup(), notebookCR.getVersion(),
              namespace, notebookCR.getPlural(), notebookCR, "true");
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_CREATE);

      // create Traefik custom resource
      createIngressRoute(notebookCR.getMetadata().getNamespace(), notebookCR.getMetadata().getName());

    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse Notebook object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse Notebook object failed by " +
          e.getMessage());
    }
    return notebook;
  }

  @Override
  public Notebook findNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    Notebook notebook;
    String namespace = "default";
    
    if (System.getenv(ENV_NAMESPACE) != null) {
      namespace = System.getenv(ENV_NAMESPACE);
    }
    
    try {
           
        
      NotebookCR notebookCR = NotebookSpecParser.parseNotebook(spec);
      Object object = api.getNamespacedCustomObject(notebookCR.getGroup(), notebookCR.getVersion(),
              namespace,
              notebookCR.getPlural(), notebookCR.getMetadata().getName());
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return notebook;
  }

  @Override
  public Notebook deleteNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    Notebook notebook;
    final String name = spec.getMeta().getName();
    final String pvName = NotebookUtils.PV_PREFIX + name;
    final String pvcName = NotebookUtils.PVC_PREFIX + name;
    String namespace = "default";
    
    if (System.getenv(ENV_NAMESPACE) != null) {
      namespace = System.getenv(ENV_NAMESPACE);
    }
    
    try {
      NotebookCR notebookCR = NotebookSpecParser.parseNotebook(spec);
      Object object = api.deleteNamespacedCustomObject(notebookCR.getGroup(), notebookCR.getVersion(),
              namespace, notebookCR.getPlural(),
              notebookCR.getMetadata().getName(),
              new V1DeleteOptionsBuilder().withApiVersion(notebookCR.getApiVersion()).build(),
              null, null, null);
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_DELETE);
      deleteIngressRoute(namespace, notebookCR.getMetadata().getName());
      deletePersistentVolumeClaim(pvcName, namespace);
      deletePersistentVolume(pvName);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return notebook;
  }

  @Override
  public List<Notebook> listNotebook(String id) throws SubmarineRuntimeException {
    List<Notebook> notebookList;
    try {
      Object object = api.listClusterCustomObject(NotebookCR.CRD_NOTEBOOK_GROUP_V1,
              NotebookCR.CRD_NOTEBOOK_VERSION_V1, NotebookCR.CRD_NOTEBOOK_PLURAL_V1,
              "true", null, NotebookCR.NOTEBOOK_OWNER_SELECTOR_KET + "=" + id,
              null, null, null);
      notebookList = NotebookUtils.parseObjectForList(object);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return notebookList;
  }

  @Override
  public Serve createServe(String modelName, String modelVersion, String namespace) 
      throws SubmarineRuntimeException {
    ServeSpecParser parser = new ServeSpecParser(modelName, modelVersion, namespace);
    V1Deployment deployment = parser.getDeployment();
    V1Service svc = parser.getService();
    IngressRoute ingressRoute = parser.getIngressRoute();
    Middlewares middleware = parser.getMiddlewares();
    Serve serveInfo = new Serve().url(parser.getRoutePath());

    try {
      appsV1Api.createNamespacedDeployment(namespace, deployment, "true", null, null);
      coreApi.createNamespacedService(namespace, svc, "true", null, null);

      api.createNamespacedCustomObject(
            middleware.getGroup(), middleware.getVersion(),
            middleware.getMetadata().getNamespace(),
            middleware.getPlural(), middleware, "true");
            
      api.createNamespacedCustomObject(
            ingressRoute.getGroup(), ingressRoute.getVersion(),
            ingressRoute.getMetadata().getNamespace(),
            ingressRoute.getPlural(), ingressRoute, "true");
      return serveInfo;
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Serve deleteServe(String modelName, String modelVersion, String namespace) 
      throws SubmarineRuntimeException {
    ServeSpecParser parser = new ServeSpecParser(modelName, modelVersion, namespace);
    IngressRoute ingressRoute = parser.getIngressRoute();
    Middlewares middleware = parser.getMiddlewares();
    Serve serveInfo = new Serve().url(parser.getRoutePath());

    try {
      appsV1Api.deleteNamespacedDeployment(parser.getGeneralName(), namespace, "true",
          null, null, null, null, null);
      coreApi.deleteNamespacedService(parser.getSvcName(), namespace, "true",
          null, null, null, null, null);
      api.deleteNamespacedCustomObject(
          middleware.getGroup(), middleware.getVersion(),
          middleware.getMetadata().getNamespace(), middleware.getPlural(), parser.getMiddlewareName(),
          new V1DeleteOptionsBuilder().withApiVersion(middleware.getApiVersion()).build(),
          null, null, null);
      api.deleteNamespacedCustomObject(
          ingressRoute.getGroup(), ingressRoute.getVersion(),
          ingressRoute.getMetadata().getNamespace(), ingressRoute.getPlural(), parser.getRouteName(),
          new V1DeleteOptionsBuilder().withApiVersion(ingressRoute.getApiVersion()).build(),
          null, null, null);
      return serveInfo;
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  public void createPersistentVolume(String pvName, String hostPath, String storage) throws ApiException {
    V1PersistentVolume pv = VolumeSpecParser.parsePersistentVolume(pvName, hostPath, storage);

    try {
      V1PersistentVolume result = coreApi.createPersistentVolume(pv, "true", null, null);
    } catch (ApiException e) {
      LOG.error("Exception when creating persistent volume " + e.getMessage(), e);
      throw e;
    }
  }

  public void deletePersistentVolume(String pvName) throws ApiException {
    /*
    This version of Kubernetes-client/java has bug here.
    It will trigger exception as in https://github.com/kubernetes-client/java/issues/86
    but it can still work fine and delete the PV.
    */
    try {
      V1Status result = coreApi.deletePersistentVolume(
              pvName, "true", null,
              null, null, null, null
      );
    } catch (ApiException e) {
      LOG.error("Exception when deleting persistent volume " + e.getMessage(), e);
      throw e;
    } catch (JsonSyntaxException e) {
      if (e.getCause() instanceof IllegalStateException) {
        IllegalStateException ise = (IllegalStateException) e.getCause();
        if (ise.getMessage() != null && ise.getMessage().contains("Expected a string but was BEGIN_OBJECT")) {
          LOG.debug("Catching exception because of issue " +
            "https://github.com/kubernetes-client/java/issues/86", e);
        } else {
          throw e;
        }
      } else {
        throw e;
      }
    }
  }

  public void createPersistentVolumeClaim(String pvcName, String namespace, String volume, String storage)
          throws ApiException {
    V1PersistentVolumeClaim pvc = VolumeSpecParser.parsePersistentVolumeClaim(pvcName, volume, storage);

    try {
      V1PersistentVolumeClaim result = coreApi.createNamespacedPersistentVolumeClaim(
              namespace, pvc, "true", null, null
      );
    } catch (ApiException e) {
      LOG.error("Exception when creating persistent volume claim " + e.getMessage(), e);
      throw e;
    }
  }

  public void deletePersistentVolumeClaim(String pvcName, String namespace) throws ApiException {
    /*
    This version of Kubernetes-client/java has bug here.
    It will trigger exception as in https://github.com/kubernetes-client/java/issues/86
    but it can still work fine and delete the PVC
    */
    try {
      V1Status result = coreApi.deleteNamespacedPersistentVolumeClaim(
                pvcName, namespace, "true",
          null, null, null,
            null, null
      );
    } catch (ApiException e) {
      LOG.error("Exception when deleting persistent volume claim " + e.getMessage(), e);
      throw e;
    } catch (JsonSyntaxException e) {
      if (e.getCause() instanceof IllegalStateException) {
        IllegalStateException ise = (IllegalStateException) e.getCause();
        if (ise.getMessage() != null && ise.getMessage().contains("Expected a string but was BEGIN_OBJECT")) {
          LOG.debug("Catching exception because of issue " +
            "https://github.com/kubernetes-client/java/issues/86", e);
        } else {
          throw e;
        }
      } else {
        throw e;
      }
    }
  }

  private String getJobLabelSelector(ExperimentSpec experimentSpec) {
    // TODO(JohnTing): SELECTOR_KEY should be obtained from individual models in MLJOB
    if (experimentSpec.getMeta().getFramework()
        .equalsIgnoreCase(ExperimentMeta.SupportedMLFramework.TENSORFLOW.getName())) {
      return TF_JOB_SELECTOR_KEY + experimentSpec.getMeta().getName();
    } else {
      return PYTORCH_JOB_SELECTOR_KEY + experimentSpec.getMeta().getName();
    }
  }

  private void createIngressRoute(String namespace, String name) {
    try {
      IngressRoute ingressRoute = new IngressRoute();
      V1ObjectMeta meta = new V1ObjectMeta();
      meta.setName(name);
      meta.setNamespace(namespace);
      ingressRoute.setMetadata(meta);
      ingressRoute.setSpec(parseIngressRouteSpec(meta.getNamespace(), meta.getName()));
      api.createNamespacedCustomObject(
              ingressRoute.getGroup(), ingressRoute.getVersion(),
              ingressRoute.getMetadata().getNamespace(),
              ingressRoute.getPlural(), ingressRoute, "true");
    } catch (ApiException e) {
      LOG.error("K8s submitter: Create Traefik custom resource object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    }
  }

  private void deleteIngressRoute(String namespace, String name) {
    try {
      api.deleteNamespacedCustomObject(
              IngressRoute.CRD_INGRESSROUTE_GROUP_V1, IngressRoute.CRD_INGRESSROUTE_VERSION_V1,
              namespace, IngressRoute.CRD_INGRESSROUTE_PLURAL_V1, name,
              new V1DeleteOptionsBuilder().withApiVersion(IngressRoute.CRD_APIVERSION_V1).build(),
              null, null, null);
    } catch (ApiException e) {
      LOG.error("K8s submitter: Delete Traefik custom resource object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  private IngressRouteSpec parseIngressRouteSpec(String namespace, String name) {
    IngressRouteSpec spec = new IngressRouteSpec();
    Set<String> entryPoints = new HashSet<>();
    entryPoints.add("web");
    spec.setEntryPoints(entryPoints);

    SpecRoute route = new SpecRoute();
    route.setKind("Rule");
    route.setMatch("PathPrefix(`/notebook/" + namespace + "/" + name + "/`)");
    Set<Map<String, Object>> serviceMap = new HashSet<>();
    Map<String, Object> service = new HashMap<>();
    service.put("name", name);
    service.put("port", 80);
    serviceMap.add(service);
    route.setServices(serviceMap);
    Set<SpecRoute> routes = new HashSet<>();
    routes.add(route);
    spec.setRoutes(routes);
    return spec;
  }

  private enum ParseOp {
    PARSE_OP_RESULT,
    PARSE_OP_DELETE
  }

  public void createServe(String modelPath) {
    
  }
}
