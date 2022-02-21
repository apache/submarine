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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import okhttp3.OkHttpClient;
import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.JSON;
import io.kubernetes.client.openapi.apis.AppsV1Api;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.apis.CustomObjectsApi;
import io.kubernetes.client.custom.V1Patch;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.openapi.models.CoreV1EventList;
import io.kubernetes.client.openapi.models.V1ConfigMap;
import io.kubernetes.client.openapi.models.V1ConfigMapList;
import io.kubernetes.client.openapi.models.V1Deployment;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaim;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaimList;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.openapi.models.V1Status;
import io.kubernetes.client.util.generic.options.CreateOptions;
import io.kubernetes.client.util.generic.options.DeleteOptions;
import io.kubernetes.client.util.generic.options.ListOptions;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import io.kubernetes.client.util.generic.GenericKubernetesApi;

import org.apache.commons.lang3.StringUtils;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.serve.istio.IstioVirtualService;
import org.apache.submarine.serve.istio.IstioVirtualServiceList;
import org.apache.submarine.serve.pytorch.SeldonPytorchServing;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonDeploymentList;
import org.apache.submarine.serve.tensorflow.SeldonTFServing;
import org.apache.submarine.server.k8s.utils.K8sUtils;
import org.apache.submarine.serve.utils.IstioConstants;
import org.apache.submarine.serve.utils.SeldonConstants;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.experiment.Info;
import org.apache.submarine.server.api.experiment.MlflowInfo;
import org.apache.submarine.server.api.experiment.TensorboardInfo;
import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.NotebookCRList;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobList;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobList;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRoute;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRouteSpec;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRouteList;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.SpecRoute;
import org.apache.submarine.server.submitter.k8s.parser.ConfigmapSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.ExperimentSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.NotebookSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.VolumeSpecParser;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.apache.submarine.server.submitter.k8s.util.NotebookUtils;
import org.apache.submarine.server.submitter.k8s.util.OwnerReferenceUtils;


import org.joda.time.DateTime;
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

  // Add an exception Consumer, handle the problem that delete operation does not have the resource
  public static final Function<ApiException, Object> API_EXCEPTION_404_CONSUMER = e -> {
    if (e.getCode() != 404) {
      LOG.error("When submit resource to k8s get ApiException with code " + e.getCode(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    } else {
      return null;
    }
  };

  private static final String OVERWRITE_JSON;

  static {
    final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    OVERWRITE_JSON = conf.getString(
            SubmarineConfVars.ConfVars.SUBMARINE_NOTEBOOK_DEFAULT_OVERWRITE_JSON);
  }


  // K8s API client for CRD
  private CustomObjectsApi api;

  private GenericKubernetesApi<V1Pod, V1PodList> podClient;

  private GenericKubernetesApi<CoreV1Event, CoreV1EventList> eventClient;

  private GenericKubernetesApi<V1PersistentVolumeClaim, V1PersistentVolumeClaimList>
          persistentVolumeClaimClient;

  private GenericKubernetesApi<V1ConfigMap, V1ConfigMapList> configMapClient;

  private GenericKubernetesApi<TFJob, TFJobList> tfJobClient;

  private GenericKubernetesApi<PyTorchJob, PyTorchJobList> pyTorchJobClient;

  private GenericKubernetesApi<NotebookCR, NotebookCRList> notebookCRClient;

  private GenericKubernetesApi<IngressRoute, IngressRouteList> ingressRouteClient;

  private GenericKubernetesApi<SeldonDeployment, SeldonDeploymentList> seldonDeploymentClient;

  private GenericKubernetesApi<IstioVirtualService, IstioVirtualServiceList> istioVirtualServiceClient;

  private CoreV1Api coreApi;

  private AppsV1Api appsV1Api;

  private ApiClient client = null;

  public K8sSubmitter() {
  }

  @Override
  public void initialize(SubmarineConfiguration conf) {
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
      OkHttpClient httpClient = client.getHttpClient();
      client.setHttpClient(httpClient);
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

    podClient =
            new GenericKubernetesApi<>(
                    V1Pod.class, V1PodList.class,
                    "", "v1", "pods", client);

    eventClient =
            new GenericKubernetesApi<>(
                    CoreV1Event.class, CoreV1EventList.class,
                    "", "v1", "events", client);

    persistentVolumeClaimClient =
            new GenericKubernetesApi<>(
                    V1PersistentVolumeClaim.class, V1PersistentVolumeClaimList.class,
                    "", "v1", "persistentvolumeclaims", client);

    configMapClient =
            new GenericKubernetesApi<>(
                    V1ConfigMap.class, V1ConfigMapList.class,
                    "", "v1", "configmaps", client);

    tfJobClient =
            new GenericKubernetesApi<>(
                    TFJob.class, TFJobList.class,
                    TFJob.CRD_TF_GROUP_V1, TFJob.CRD_TF_VERSION_V1,
                    TFJob.CRD_TF_PLURAL_V1, client);

    pyTorchJobClient =
            new GenericKubernetesApi<>(
                    PyTorchJob.class, PyTorchJobList.class,
                    PyTorchJob.CRD_PYTORCH_GROUP_V1, PyTorchJob.CRD_PYTORCH_VERSION_V1,
                    PyTorchJob.CRD_PYTORCH_PLURAL_V1, client);

    notebookCRClient =
            new GenericKubernetesApi<>(
                    NotebookCR.class, NotebookCRList.class,
                    NotebookCR.CRD_NOTEBOOK_GROUP_V1, NotebookCR.CRD_NOTEBOOK_VERSION_V1,
                    NotebookCR.CRD_NOTEBOOK_PLURAL_V1, client);

    ingressRouteClient =
            new GenericKubernetesApi<>(
                    IngressRoute.class, IngressRouteList.class,
                    IngressRoute.CRD_INGRESSROUTE_GROUP_V1, IngressRoute.CRD_INGRESSROUTE_VERSION_V1,
                    IngressRoute.CRD_INGRESSROUTE_PLURAL_V1, client);

    seldonDeploymentClient =
            new GenericKubernetesApi<>(
                   SeldonDeployment.class, SeldonDeploymentList.class,
                    SeldonConstants.GROUP, SeldonConstants.VERSION,
                    SeldonConstants.PLURAL, client);

    istioVirtualServiceClient =
            new GenericKubernetesApi<>(
                    IstioVirtualService.class, IstioVirtualServiceList.class,
                    IstioConstants.GROUP, IstioConstants.VERSION,
                    IstioConstants.PLURAL, client);
  }

  @Override
  public Experiment createExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    Experiment experiment;
    try {
      MLJob mlJob = ExperimentSpecParser.parseJob(spec);
      mlJob.getMetadata().setNamespace(getServerNamespace());
      mlJob.getMetadata().setOwnerReferences(OwnerReferenceUtils.getOwnerReference());

      Object object = mlJob.getPlural().equals(TFJob.CRD_TF_PLURAL_V1)
              ? tfJobClient.create(getServerNamespace(), (TFJob) mlJob,
                      new CreateOptions()).throwsApiException().getObject()
              : pyTorchJobClient.create(getServerNamespace(), (PyTorchJob) mlJob,
                      new CreateOptions()).throwsApiException().getObject();
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
      mlJob.getMetadata().setNamespace(getServerNamespace());
      Object object = mlJob.getPlural().equals(TFJob.CRD_TF_PLURAL_V1)
              ? tfJobClient.get(getServerNamespace(), mlJob.getMetadata().getName())
                      .throwsApiException().getObject()
              : pyTorchJobClient.get(getServerNamespace(), mlJob.getMetadata().getName())
                      .throwsApiException().getObject();
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
      mlJob.getMetadata().setNamespace(getServerNamespace());

      Object object = mlJob.getPlural().equals(TFJob.CRD_TF_PLURAL_V1)
              ? tfJobClient.patch(getServerNamespace(), mlJob.getMetadata().getName(),
              V1Patch.PATCH_FORMAT_JSON_PATCH,
              new V1Patch(new Gson().toJson(((TFJob) mlJob).getSpec()))).throwsApiException().getObject()
              : pyTorchJobClient.patch(getServerNamespace(), mlJob.getMetadata().getName(),
              V1Patch.PATCH_FORMAT_JSON_PATCH,
              new V1Patch(new Gson().toJson(((PyTorchJob) mlJob).getSpec()))).throwsApiException().getObject()
              ;
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
      mlJob.getMetadata().setNamespace(getServerNamespace());

      Object object = mlJob.getPlural().equals(TFJob.CRD_TF_PLURAL_V1)
              ? tfJobClient.delete(getServerNamespace(), mlJob.getMetadata().getName(),
              MLJobConverter.toDeleteOptionsFromMLJob(mlJob))
              .throwsApiException().getStatus()
              : pyTorchJobClient.delete(getServerNamespace(), mlJob.getMetadata().getName(),
              MLJobConverter.toDeleteOptionsFromMLJob(mlJob))
              .throwsApiException().getStatus();
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
      ListOptions listOptions = new ListOptions();
      listOptions.setLabelSelector(getJobLabelSelector(spec));
      final V1PodList podList = podClient.list(getServerNamespace(), listOptions)
              .throwsApiException().getObject();
      for (V1Pod pod : podList.getItems()) {
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
      ListOptions listOptions = new ListOptions();
      listOptions.setLabelSelector(getJobLabelSelector(spec));
      final V1PodList podList = podClient.list(getServerNamespace(), listOptions)
              .throwsApiException().getObject();
      for (V1Pod pod : podList.getItems()) {
        String podName = pod.getMetadata().getName();
        String podLog = coreApi.readNamespacedPodLog(
            podName, getServerNamespace(), null, Boolean.FALSE, null,
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
    try {
      return new TensorboardInfo(getInfo(name, ingressRouteName));
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public MlflowInfo getMlflowInfo() throws SubmarineRuntimeException {
    final String name = "submarine-mlflow";
    final String ingressRouteName = "submarine-mlflow-ingressroute";
    try {
      return new MlflowInfo(getInfo(name, ingressRouteName));
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  public Info getInfo(String name, String ingressRouteName) throws ApiException{
    V1Deployment deploy = appsV1Api.readNamespacedDeploymentStatus(name, getServerNamespace(), "true");
    boolean available = deploy.getStatus().getAvailableReplicas() > 0; // at least one replica is running

    IngressRoute ingressRoute = new IngressRoute();
    V1ObjectMeta meta = new V1ObjectMeta();
    meta.setName(ingressRouteName);
    meta.setNamespace(getServerNamespace());
    ingressRoute.setMetadata(meta);

    IngressRoute result = ingressRouteClient.get(getServerNamespace(), ingressRouteName)
            .throwsApiException().getObject();

    String route = result.getSpec().getRoutes().stream().findFirst().get().getMatch();

    String url = route.replace("PathPrefix(`", "").replace("`)", "/");

    return new Info(available, url);
  }
  @Override
  public Notebook createNotebook(NotebookSpec spec, String notebookId) throws SubmarineRuntimeException {
    Notebook notebook;
    final String name = spec.getMeta().getName();
    final String scName = NotebookUtils.SC_NAME;
    final String host = NotebookUtils.HOST_PATH;
    final String workspacePvc = String.format("%s-%s", NotebookUtils.PVC_PREFIX, name);
    final String userPvc = String.format("%s-user-%s", NotebookUtils.PVC_PREFIX, name);
    final String configmap = String.format("%s-%s", NotebookUtils.OVERWRITE_PREFIX, name);
    String namespace = getServerNamespace();

    // parse notebook custom resource
    NotebookCR notebookCR;
    try {
      notebookCR = NotebookSpecParser.parseNotebook(spec, notebookId, namespace);
      notebookCR.getMetadata().setNamespace(namespace);
      notebookCR.getMetadata().setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    }

    // create persistent volume claim
    try {
      // workspace
      createPersistentVolumeClaim(workspacePvc, namespace, scName, NotebookUtils.STORAGE);
      // user setting
      createPersistentVolumeClaim(userPvc, namespace, scName, NotebookUtils.DEFAULT_USER_STORAGE);
    } catch (ApiException e) {
      LOG.error("K8s submitter: Create persistent volume claim for Notebook object failed by " +
          e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: Create persistent volume claim for " +
          "Notebook object failed by " + e.getMessage());
    }

    // create configmap if needed
    boolean needOverwrite = StringUtils.isNotBlank(OVERWRITE_JSON);
    if (needOverwrite) {
      try {
        createConfigMap(configmap, namespace, NotebookUtils.DEFAULT_OVERWRITE_FILE_NAME, OVERWRITE_JSON);
      } catch (JsonSyntaxException e) {
        LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
        rollbackCreationPVC(namespace, workspacePvc, userPvc);
        throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
      } catch (ApiException e) {
        LOG.error("K8s submitter: parse Notebook object failed by " + e.getMessage(), e);
        rollbackCreationPVC(namespace, workspacePvc, userPvc);
        throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse Notebook object failed by " +
                e.getMessage());
      }
    }

    // create notebook custom resource
    try {
      Object object = notebookCRClient.create(notebookCR).throwsApiException().getObject();
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_CREATE);
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      if (needOverwrite) rollbackCreationConfigMap(namespace, configmap);
      rollbackCreationPVC(namespace, workspacePvc, userPvc);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse Notebook object failed by " + e.getMessage(), e);
      if (needOverwrite) rollbackCreationConfigMap(namespace, configmap);
      rollbackCreationPVC(namespace, workspacePvc, userPvc);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse Notebook object failed by " +
              e.getMessage());
    }

    // create notebook Traefik custom resource
    try {
      createIngressRoute(notebookCR.getMetadata().getNamespace(), notebookCR.getMetadata().getName());
    } catch (ApiException e) {
      LOG.error("K8s submitter: Create ingressroute for Notebook object failed by " +
          e.getMessage(), e);
      rollbackCreationNotebook(notebookCR, namespace);
      if (needOverwrite) rollbackCreationConfigMap(namespace, configmap);
      rollbackCreationPVC(namespace, workspacePvc, userPvc);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: ingressroute for Notebook " +
          "object failed by " + e.getMessage());
    }
    return notebook;
  }

  @Override
  public Notebook findNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    Notebook notebook = null;
    String namespace = getServerNamespace();

    try {
      NotebookCR notebookCR = NotebookSpecParser.parseNotebook(spec, null, null);
      Object object = notebookCRClient.get(namespace, notebookCR.getMetadata().getName())
              .throwsApiException().getObject();
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
      if (notebook.getStatus().equals(Notebook.Status.STATUS_WAITING.toString())) {
        LOG.info(String.format("notebook status: waiting; check the pods in namespace:[%s] to "
            + "ensure is the waiting caused by image pulling", namespace));
        String podLabelSelector = String.format("%s=%s", NotebookCR.NOTEBOOK_ID,
            spec.getMeta().getLabels().get(NotebookCR.NOTEBOOK_ID).toString());
        ListOptions listOptions = new ListOptions();
        listOptions.setLabelSelector(podLabelSelector);
        final V1PodList podList = podClient.list(getServerNamespace(), listOptions)
                .throwsApiException().getObject();
        String podName = podList.getItems().get(0).getMetadata().getName();

        String fieldSelector = String.format("involvedObject.name=%s", podName);
        listOptions = new ListOptions();
        listOptions.setFieldSelector(fieldSelector);
        CoreV1EventList events = eventClient.list(namespace, listOptions).throwsApiException().getObject();
        CoreV1Event latestEvent = events.getItems().get(events.getItems().size() - 1);

        if (latestEvent.getReason().equalsIgnoreCase("Pulling")) {
          notebook.setStatus(Notebook.Status.STATUS_PULLING.getValue());
          notebook.setReason(latestEvent.getReason());
        }
      }
    } catch (ApiException e) {
      // SUBMARINE-1124
      // The exception that obtaining CRD resources is not necessarily because the CRD is deleted,
      // but maybe due to timeout or API error caused by network and other reasons.
      // Therefore, the status of the notebook should be set to a new enum NOTFOUND.
      LOG.warn("Get error when submitter is finding notebook: {}", spec.getMeta().getName());
      if (notebook == null) {
        notebook = new Notebook();
      }
      notebook.setName(spec.getMeta().getName());
      notebook.setSpec(spec);
      notebook.setReason(e.getMessage());
      notebook.setStatus(Notebook.Status.STATUS_NOT_FOUND.getValue());
    }
    return notebook;
  }

  @Override
  public Notebook deleteNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    Notebook notebook = null;
    final String name = spec.getMeta().getName();
    String namespace = getServerNamespace();
    NotebookCR notebookCR = NotebookSpecParser.parseNotebook(spec, null, null);
    try {
      Object object = notebookCRClient.delete(namespace, name,
              getDeleteOptions(notebookCR.getApiVersion())).throwsApiException().getStatus();
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_DELETE);
    } catch (ApiException e) {
      API_EXCEPTION_404_CONSUMER.apply(e);
    } finally {
      if (notebook == null) {
        // add metadata time info
        notebookCR.getMetadata().setDeletionTimestamp(new DateTime());
        // build notebook response
        notebook = NotebookUtils.buildNotebookResponse(notebookCR);
        notebook.setStatus(Notebook.Status.STATUS_NOT_FOUND.getValue());
        notebook.setReason("The notebook instance is not found");
      }
    }

    // delete ingress route
    deleteIngressRoute(namespace, name);

    // delete pvc
    // workspace pvc
    deletePersistentVolumeClaim(String.format("%s-%s", NotebookUtils.PVC_PREFIX, name), namespace);
    // user set pvc
    deletePersistentVolumeClaim(String.format("%s-user-%s", NotebookUtils.PVC_PREFIX, name), namespace);

    // configmap
    if (StringUtils.isNoneBlank(OVERWRITE_JSON)) {
      deleteConfigMap(namespace, String.format("%s-%s", NotebookUtils.OVERWRITE_PREFIX, name));
    }

    return notebook;
  }

  @Override
  public List<Notebook> listNotebook(String id) throws SubmarineRuntimeException {
    List<Notebook> notebookList;
    String namespace = getServerNamespace();

    try {
      ListOptions listOptions = new ListOptions();
      listOptions.setLabelSelector(NotebookCR.NOTEBOOK_OWNER_SELECTOR_KEY + "=" + id);
      Object object = notebookCRClient.list(namespace, listOptions).throwsApiException().getObject();
      notebookList = NotebookUtils.parseObjectForList(object);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return notebookList;
  }

  @Override
  public void createServe(ServeSpec spec)
      throws SubmarineRuntimeException {
    SeldonDeployment seldonDeployment = parseServeSpec(spec);
    IstioVirtualService istioVirtualService = new IstioVirtualService(spec.getModelName(),
        spec.getModelVersion());
    try {
      seldonDeploymentClient.create("default", seldonDeployment, new CreateOptions()).throwsApiException();
    } catch (ApiException e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    try {
      istioVirtualServiceClient.create("default", istioVirtualService, new CreateOptions())
              .throwsApiException();
    } catch (ApiException e) {
      LOG.error(e.getMessage(), e);
      try {
        seldonDeploymentClient.delete("default", seldonDeployment.getMetadata().getName(),
            getDeleteOptions(seldonDeployment.getApiVersion())).throwsApiException();
      } catch (ApiException e1) {
        LOG.error(e1.getMessage(), e1);
      }
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public void deleteServe(ServeSpec spec)
      throws SubmarineRuntimeException {
    SeldonDeployment seldonDeployment = parseServeSpec(spec);
    IstioVirtualService istioVirtualService = new IstioVirtualService(spec.getModelName(),
        spec.getModelVersion());
    try {
      seldonDeploymentClient.delete("default", seldonDeployment.getMetadata().getName(),
              getDeleteOptions(seldonDeployment.getApiVersion())).throwsApiException();
      istioVirtualServiceClient.delete("default", istioVirtualService.getMetadata().getName(),
              getDeleteOptions(istioVirtualService.getApiVersion())).throwsApiException();
    } catch (ApiException e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  public void createPersistentVolumeClaim(String pvcName, String namespace, String scName, String storage)
      throws ApiException {
    V1PersistentVolumeClaim pvc = VolumeSpecParser.parsePersistentVolumeClaim(pvcName, scName, storage);
    pvc.getMetadata().setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
    try {
      persistentVolumeClaimClient.create(namespace, pvc, new CreateOptions()).throwsApiException();
    } catch (ApiException e) {
      LOG.error("Exception when creating persistent volume claim " + e.getMessage(), e);
      throw e;
    }
  }

  public void deletePersistentVolumeClaim(String pvcName, String namespace) {
    /*
    This version of Kubernetes-client/java has bug here.
    It will trigger exception as in https://github.com/kubernetes-client/java/issues/86
    but it can still work fine and delete the PVC
    */
    try {
      persistentVolumeClaimClient.delete(namespace, pvcName).throwsApiException();
    } catch (ApiException e) {
      LOG.error("Exception when deleting persistent volume claim " + e.getMessage(), e);
      API_EXCEPTION_404_CONSUMER.apply(e);
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
    if (experimentSpec.getMeta().getFramework()
        .equalsIgnoreCase(ExperimentMeta.SupportedMLFramework.TENSORFLOW.getName())) {
      return TF_JOB_SELECTOR_KEY + experimentSpec.getMeta().getExperimentId();
    } else {
      return PYTORCH_JOB_SELECTOR_KEY + experimentSpec.getMeta().getExperimentId();
    }
  }

  private void createIngressRoute(String namespace, String name) throws ApiException {
    try {
      IngressRoute ingressRoute = new IngressRoute();
      V1ObjectMeta meta = new V1ObjectMeta();
      meta.setName(name);
      meta.setNamespace(namespace);
      meta.setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
      ingressRoute.setMetadata(meta);
      ingressRoute.setSpec(parseIngressRouteSpec(meta.getNamespace(), meta.getName()));

      ingressRouteClient.create(namespace, ingressRoute, new CreateOptions()).throwsApiException();
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
      ingressRouteClient.delete(namespace, name,
          getDeleteOptions(IngressRoute.CRD_APIVERSION_V1)).throwsApiException();
    } catch (ApiException e) {
      LOG.error("K8s submitter: Delete Traefik custom resource object failed by " + e.getMessage(), e);
      API_EXCEPTION_404_CONSUMER.apply(e);
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

  private SeldonDeployment parseServeSpec(ServeSpec spec) throws SubmarineRuntimeException {
    String modelName = spec.getModelName();
    String modelType = spec.getModelType();
    String modelURI = spec.getModelURI();

    SeldonDeployment seldonDeployment;
    if (modelType.equals("tensorflow")){
      seldonDeployment = new SeldonTFServing(modelName, modelURI);
    } else if (modelType.equals("pytorch")){
      seldonDeployment = new SeldonPytorchServing(modelName, modelURI);
    } else {
      throw new SubmarineRuntimeException("Given serve type: " + modelType + " is not supported.");
    }
    return seldonDeployment;
  }

  /**
   * Create ConfigMap with values (key1, value1, key2, value2, ...)
   */
  public void createConfigMap(String name, String namespace, String ... values)
          throws ApiException {
    V1ConfigMap configMap = ConfigmapSpecParser.parseConfigMap(name, values);
    configMap.getMetadata().setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
    try {
      configMapClient.create(namespace, configMap, new CreateOptions()).throwsApiException();
    } catch (ApiException e) {
      LOG.error("Exception when creating configmap " + e.getMessage(), e);
      throw e;
    }
  }

  /**
   * Delete ConfigMap
   */
  public void deleteConfigMap(String namespace, String name) {
    try {
      configMapClient.delete(namespace, name).throwsApiException();
    } catch (ApiException e) {
      LOG.error("Exception when deleting config map " + e.getMessage(), e);
      API_EXCEPTION_404_CONSUMER.apply(e);
    }
  }

  /**
   * Rollback to delete ConfigMap
   */
  private void rollbackCreationConfigMap(String namespace, String ... names)
          throws SubmarineRuntimeException {
    for (String name : names) {
      deleteConfigMap(namespace, name);
    }
  }

  private void rollbackCreationPVC(String namespace, String ... pvcNames) {
    for (String pvcName : pvcNames) {
      deletePersistentVolumeClaim(pvcName, namespace);
    }
  }

  private void rollbackCreationNotebook(NotebookCR notebookCR, String namespace)
      throws SubmarineRuntimeException {
    try {
      notebookCRClient.delete(namespace, notebookCR.getMetadata().getName(),
              getDeleteOptions(notebookCR.getApiVersion())).throwsApiException();
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  private String getServerNamespace() {
    return K8sUtils.getNamespace();
  }

  private DeleteOptions getDeleteOptions(String apiVersion){
    DeleteOptions deleteOptions = new DeleteOptions();
    deleteOptions.setApiVersion(apiVersion);
    return deleteOptions;
  }

  private enum ParseOp {
    PARSE_OP_RESULT,
    PARSE_OP_DELETE
  }
}
