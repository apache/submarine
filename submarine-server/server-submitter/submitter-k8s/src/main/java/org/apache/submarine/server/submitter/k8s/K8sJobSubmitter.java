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

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import io.kubernetes.client.ApiClient;
import io.kubernetes.client.ApiException;
import io.kubernetes.client.Configuration;
import io.kubernetes.client.JSON;
import io.kubernetes.client.apis.CoreV1Api;
import io.kubernetes.client.apis.CustomObjectsApi;
import io.kubernetes.client.models.V1Pod;
import io.kubernetes.client.models.V1PodList;
import io.kubernetes.client.models.V1Status;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.job.JobSubmitter;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.job.JobLog;
import org.apache.submarine.server.api.spec.JobLibrarySpec;
import org.apache.submarine.server.api.spec.JobSpec;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.apache.submarine.server.submitter.k8s.parser.JobSpecParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JobSubmitter for Kubernetes Cluster.
 */
public class K8sJobSubmitter implements JobSubmitter {
  private static final Logger LOG = LoggerFactory.getLogger(K8sJobSubmitter.class);

  private static final String KUBECONFIG_ENV = "KUBECONFIG";

  private static final String TF_JOB_SELECTOR_KEY = "tf-job-name=";
  private static final String PYTORCH_JOB_SELECTOR_KEY = "pytorch-job-name=";

  // K8s API client for CRD
  private CustomObjectsApi api;

  private CoreV1Api coreApi;

  public K8sJobSubmitter() {}

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
  }

  @Override
  public String getSubmitterType() {
    return "k8s";
  }

  @Override
  public Job createJob(JobSpec jobSpec) throws SubmarineRuntimeException {
    Job job;
    try {
      MLJob mlJob = JobSpecParser.parseJob(jobSpec);
      Object object = api.createNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob, "true");
      job = parseResponseObject(object, ParseOp.PARSE_OP_RESULT);
    } catch (InvalidSpecException e) {
      LOG.error("K8s submitter: parse Job object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse Job object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return job;
  }

  @Override
  public Job findJob(JobSpec jobSpec) throws SubmarineRuntimeException {
    Job job;
    try {
      MLJob mlJob = JobSpecParser.parseJob(jobSpec);
      Object object = api.getNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob.getMetadata().getName());
      job = parseResponseObject(object, ParseOp.PARSE_OP_RESULT);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return job;
  }

  @Override
  public Job patchJob(JobSpec jobSpec) throws SubmarineRuntimeException {
    Job job;
    try {
      MLJob mlJob = JobSpecParser.parseJob(jobSpec);
      Object object = api.patchNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob.getMetadata().getName(),
          mlJob);
      job = parseResponseObject(object, ParseOp.PARSE_OP_RESULT);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return job;
  }

  @Override
  public Job deleteJob(JobSpec jobSpec) throws SubmarineRuntimeException {
    Job job;
    try {
      MLJob mlJob = JobSpecParser.parseJob(jobSpec);
      Object object = api.deleteNamespacedCustomObject(mlJob.getGroup(), mlJob.getVersion(),
          mlJob.getMetadata().getNamespace(), mlJob.getPlural(), mlJob.getMetadata().getName(),
          MLJobConverter.toDeleteOptionsFromMLJob(mlJob), null, null, null);
      job = parseResponseObject(object, ParseOp.PARSE_OP_DELETE);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return job;
  }

  private Job parseResponseObject(Object object, ParseOp op) throws SubmarineRuntimeException {
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
  public JobLog getJobLogName(JobSpec jobSpec, String jobId) {
    JobLog jobLog = new JobLog();
    jobLog.setJobId(jobId);
    try {
      final V1PodList podList = coreApi.listNamespacedPod(
          jobSpec.getNamespace(),
          "false", null, null,
          getJobLabelSelector(jobSpec), null, null,
          null, null);
      for (V1Pod pod: podList.getItems()) {
        String podName = pod.getMetadata().getName();
        jobLog.addPodLog(podName, null);
      }
    } catch (final ApiException e) {
      LOG.error("Error when listing pod for job:" + jobSpec.getName(), e.getMessage());
    }
    return jobLog;
  }

  @Override
  public JobLog getJobLog(JobSpec jobSpec, String jobId) {
    JobLog jobLog = new JobLog();
    jobLog.setJobId(jobId);
    try {
      final V1PodList podList = coreApi.listNamespacedPod(
          jobSpec.getNamespace(),
          "false", null, null,
          getJobLabelSelector(jobSpec), null, null,
          null, null);
      
      for (V1Pod pod : podList.getItems()) {
        String podName = pod.getMetadata().getName();
        String namespace = pod.getMetadata().getNamespace();
        String podLog = coreApi.readNamespacedPodLog(
            podName, namespace, null, Boolean.FALSE,
            Integer.MAX_VALUE, null, Boolean.FALSE, 
            Integer.MAX_VALUE, null, Boolean.FALSE);

        jobLog.addPodLog(podName, podLog);
      }
    } catch (final ApiException e) {
      LOG.error("Error when listing pod for job:" + jobSpec.getName(), e.getMessage());
    }
    return jobLog;
  }
  
  private String getJobLabelSelector(JobSpec jobSpec) {
    // TODO(JohnTing): SELECTOR_KEY should be obtained from individual models in MLJOB
    if (jobSpec.getLibrarySpec()
        .getName().equalsIgnoreCase(JobLibrarySpec.SupportedMLFramework.TENSORFLOW.getName())) {
      return TF_JOB_SELECTOR_KEY + jobSpec.getName();
    } else {
      return PYTORCH_JOB_SELECTOR_KEY + jobSpec.getName();
    }
  }

  private enum ParseOp {
    PARSE_OP_RESULT,
    PARSE_OP_DELETE
  }
}
