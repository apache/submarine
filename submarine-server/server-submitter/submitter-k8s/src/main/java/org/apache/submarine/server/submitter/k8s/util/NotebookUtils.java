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

package org.apache.submarine.server.submitter.k8s.util;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import io.kubernetes.client.openapi.JSON;
import io.kubernetes.client.openapi.models.V1ContainerState;
import io.kubernetes.client.openapi.models.V1Status;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.NotebookCRList;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utils for building response of k8s (notebook) submitter
 */
public class NotebookUtils {

  private static final Logger LOG = LoggerFactory.getLogger(NotebookUtils.class);
  public static final String STORAGE = "10Gi";
  public static final String DEFAULT_USER_STORAGE = "100Mi";
  public static final String SC_NAME = "submarine-storageclass";
  public static final String STORAGE_PREFIX = "notebook-storage";
  public static final String PV_PREFIX = "notebook-pv";
  public static final String PVC_PREFIX = "notebook-pvc";
  public static final String OVERWRITE_PREFIX = "overwrite-configmap";
  public static final String HOST_PATH = "/mnt";
  public static final String DEFAULT_OVERWRITE_FILE_NAME = "overrides.json";

  public enum ParseOpt {
    PARSE_OPT_CREATE,
    PARSE_OPT_GET,
    PARSE_OPT_DELETE;
  }

  public static Notebook parseObject(Object obj, ParseOpt opt) throws SubmarineRuntimeException {
    Gson gson = new JSON().getGson();
    String jsonString = gson.toJson(obj);
    LOG.info("Upstream response JSON: {}", jsonString);
    try {
      if (opt == ParseOpt.PARSE_OPT_DELETE) {
        V1Status status = gson.fromJson(jsonString, V1Status.class);
        return buildNotebookResponseFromStatus(status);
      } else {
        NotebookCR notebookCR = gson.fromJson(jsonString, NotebookCR.class);
        return buildNotebookResponse(notebookCR);
      }
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
    }
    throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
  }

  public static List<Notebook> parseObjectForList(Object object) throws SubmarineRuntimeException {
    Gson gson = new JSON().getGson();
    String jsonString = gson.toJson(object);
    LOG.info("Upstream response JSON: {}", jsonString);

    try {
      List<Notebook> notebookList = new ArrayList<>();
      NotebookCRList notebookCRList = gson.fromJson(jsonString, NotebookCRList.class);
      for (NotebookCR notebookCR : notebookCRList.getItems()) {
        Notebook notebook = buildNotebookResponse(notebookCR);
        notebookList.add(notebook);
      }
      return notebookList;
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
    }
    throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
  }

  private static Notebook buildNotebookResponse(NotebookCR notebookCR) {
    Notebook notebook = new Notebook();
    notebook.setUid(notebookCR.getMetadata().getUid());
    notebook.setName(notebookCR.getMetadata().getName());
    notebook.setCreatedTime(notebookCR.getMetadata().getCreationTimestamp().toString());
    // notebook url
    notebook.setUrl("/notebook/" + notebookCR.getMetadata().getNamespace() + "/" +
            notebookCR.getMetadata().getName() + "/lab");

    // process status
    Map<String, String> statusMap = processStatus(notebookCR);
    notebook.setStatus(statusMap.get("status"));
    notebook.setReason(statusMap.get("reason"));

    if (notebookCR.getMetadata().getDeletionTimestamp() != null) {
      notebook.setDeletedTime(notebookCR.getMetadata().getDeletionTimestamp().toString());
    }
    return notebook;
  }

  private static Map<String, String> processStatus(NotebookCR notebookCR) {
    Map<String, String> statusMap = new HashMap<>();
    // if the notebook instance is deleted
    if (notebookCR.getMetadata().getDeletionTimestamp() != null) {
      statusMap = createStatusMap(Notebook.Status.STATUS_TERMINATING.toString(),
              "The notebook instance is terminating");
    }

    if (notebookCR.getStatus() == null) {
      // if the notebook pod has not been created
      statusMap = createStatusMap(Notebook.Status.STATUS_CREATING.toString(),
              "The notebook instance is creating");
    } else {
      // if the notebook instance is ready(Running)
      int replicas = notebookCR.getStatus().getReadyReplicas();
      if (replicas == 1) {
        statusMap = createStatusMap(Notebook.Status.STATUS_RUNNING.toString(),
                "The notebook instance is running");
      }

      // if the notebook instance is waiting
      V1ContainerState containerState = notebookCR.getStatus().getContainerState();
      if (containerState.getWaiting() != null) {
        statusMap = createStatusMap(Notebook.Status.STATUS_WAITING.toString(),
                containerState.getWaiting().getReason());
      }
    }

    return statusMap;
  }

  private static Map<String, String> createStatusMap(String status, String reason) {
    Map<String, String> statusMap = new HashMap<>();
    statusMap.put("status", status);
    statusMap.put("reason", reason);
    return statusMap;
  }

  private static Notebook buildNotebookResponseFromStatus(V1Status status) {
    Notebook notebook = new Notebook();
    if (status.getStatus().toLowerCase().equals("success")) {
      notebook.setStatus(Notebook.Status.STATUS_TERMINATING.toString());
      notebook.setReason("The notebook instance is terminating");
    }

    if (status.getDetails() != null) {
      notebook.setUid(status.getDetails().getUid());
      notebook.setName(status.getDetails().getName());
    }
    return notebook;
  }
}
