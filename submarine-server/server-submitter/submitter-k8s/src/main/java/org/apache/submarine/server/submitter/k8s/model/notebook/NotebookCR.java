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

package org.apache.submarine.server.submitter.k8s.model.notebook;

import com.google.gson.JsonSyntaxException;
import com.google.gson.annotations.SerializedName;

import io.kubernetes.client.common.KubernetesObject;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.K8sSubmitter;
import org.apache.submarine.server.submitter.k8s.model.K8sResource;
import org.apache.submarine.server.submitter.k8s.parser.NotebookSpecParser;
import org.apache.submarine.server.submitter.k8s.util.NotebookUtils;
import org.apache.submarine.server.submitter.k8s.util.OwnerReferenceUtils;
import org.apache.submarine.server.submitter.k8s.util.YamlUtils;
import org.joda.time.DateTime;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

import static org.apache.submarine.server.submitter.k8s.K8sSubmitter.getDeleteOptions;

public class NotebookCR implements KubernetesObject, K8sResource<Notebook> {

  private static final Logger LOG = LoggerFactory.getLogger(NotebookCR.class);

  public static final String CRD_NOTEBOOK_VERSION_V1 = "v1";
  public static final String CRD_NOTEBOOK_GROUP_V1 = "kubeflow.org";
  public static final String CRD_NOTEBOOK_APIVERSION_V1 =
          CRD_NOTEBOOK_GROUP_V1 + "/" + CRD_NOTEBOOK_VERSION_V1;
  public static final String CRD_NOTEBOOK_KIND_V1 = "Notebook";
  public static final String CRD_NOTEBOOK_PLURAL_V1 = "notebooks";
  public static final String NOTEBOOK_OWNER_SELECTOR_KEY = "notebook-owner-id";
  public static final String NOTEBOOK_ID = "notebook-id";

  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ObjectMeta metadata;

  private transient String group;

  private transient String version;

  private transient String plural;

  @SerializedName("spec")
  private NotebookCRSpec spec;

  @SerializedName("status")
  private NotebookStatus status;

  public String getApiVersion() {
    return apiVersion;
  }

  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public String getKind() {
    return kind;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  public V1ObjectMeta getMetadata() {
    return metadata;
  }

  public void setMetadata(V1ObjectMeta metadata) {
    this.metadata = metadata;
  }

  public String getGroup() {
    return this.group;
  }

  public void setGroup(String group) {
    this.group = group;
  }

  public String getVersion() {
    return this.version;
  }

  public void setVersion(String version) {
    this.version = version;
  }

  public String getPlural() {
    return this.plural;
  }

  public void setPlural(String plural) {
    this.plural = plural;
  }

  public NotebookCRSpec getSpec() {
    return spec;
  }

  public void setSpec(NotebookCRSpec spec) {
    this.spec = spec;
  }

  public NotebookStatus getStatus() {
    return status;
  }

  public void setStatus(NotebookStatus status) {
    this.status = status;
  }

  private String notebookId;

  private NotebookSpec notebookSpec;

  public String getNotebookId() {
    return notebookId;
  }

  public NotebookSpec getNotebookSpec() {
    return notebookSpec;
  }

  public NotebookCR() {
    setApiVersion(CRD_NOTEBOOK_APIVERSION_V1);
    setKind(CRD_NOTEBOOK_KIND_V1);
    setPlural(CRD_NOTEBOOK_PLURAL_V1);
    setGroup(CRD_NOTEBOOK_GROUP_V1);
    setVersion(CRD_NOTEBOOK_VERSION_V1);
  }

  public NotebookCR(NotebookSpec notebookSpec, String notebookId, String namespace) {
    this();
    this.notebookId = notebookId;
    this.notebookSpec = notebookSpec;
    String notebookName = String.format("%s-%s", notebookId.replace("_", "-"),
            notebookSpec.getMeta().getName());
    try {
      // set metadata
      V1ObjectMeta meta = new V1ObjectMeta();
      meta.setName(notebookName);
      meta.setNamespace(namespace);
      // we need to use some labels to define/filter the properties of notebook
      Map<String, String> labels = notebookSpec.getMeta().getLabels();
      if (labels == null) {
        labels = new HashMap<>(2);
      }
      labels.put("notebook-owner-id", notebookSpec.getMeta().getOwnerId());
      labels.put("notebook-id", notebookId);
      meta.setLabels(labels);
      meta.setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
      this.setMetadata(meta);
      // set spec
      this.setSpec(NotebookSpecParser.parseNotebookCRSpec(notebookSpec, notebookName));
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    }
  }

  /**
   * Rest return notebook name.
   * The name of created notebook resource will be replaced when it is created,
   * so the name needs to be reset when it is returned.
   */
  private Notebook resetName(Notebook notebook) {
    if (getNotebookSpec() != null) {
      notebook.setName(getNotebookSpec().getMeta().getName());
    }
    return notebook;
  }

  /**
   * Create Notebook CRD
   * @param client K8sClient
   * @param tolerate Update when create conflicts
   * @return Notebook
   */
  public Notebook create(K8sClient client, boolean tolerate) {
    Notebook notebook = null;
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create Notebook resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      Object object = client.getNotebookCRClient()
          .create(this)
          .throwsApiException()
          .getObject();
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_CREATE);
    } catch (JsonSyntaxException e) {
      LOG.error("K8s submitter: parse response object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(500, "K8s Submitter parse upstream response failed.");
    } catch (ApiException e) {
      if (e.getCode() == 409 && tolerate) {// conflict
        // todo need to replace CRD
        LOG.warn("K8s submitter: resource already exists, need to replace it.", e);
        notebook = NotebookUtils.parseObject(this, NotebookUtils.ParseOpt.PARSE_OPT_REPLACE);
      } else {
        LOG.error("K8s submitter: parse Notebook object failed by " + e.getMessage(), e);
        throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse Notebook object failed by " +
                e.getMessage());
      }
    }
    return resetName(notebook);
  }

  @Override
  public Notebook create(K8sClient api) {
    return create(api, false);
  }

  @Override
  public Notebook replace(K8sClient api) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Notebook delete(K8sClient api) {
    Notebook notebook = null;
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete Notebook resource in namespace: {} and name: {}",
            this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      Object object = api.getNotebookCRClient()
          .delete(
              getMetadata().getNamespace(),
              this.getMetadata().getName(),
              getDeleteOptions(this.getApiVersion())
          ).throwsApiException().getStatus();
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_DELETE);
    } catch (ApiException e) {
      K8sSubmitter.API_EXCEPTION_404_CONSUMER.apply(e);
    } finally {
      if (notebook == null) {
        // add metadata time info
        this.getMetadata().setDeletionTimestamp(new DateTime());
        // build notebook response
        notebook = NotebookUtils.buildNotebookResponse(this);
        notebook.setStatus(Notebook.Status.STATUS_NOT_FOUND.getValue());
        notebook.setReason("The notebook instance is not found");
      }
    }
    return resetName(notebook);
  }

  @Override
  public Notebook read(K8sClient api) {
    Notebook notebook = null;
    try {
      Object object = api.getNotebookCRClient()
          .get(getMetadata().getNamespace(), getMetadata().getName())
          .throwsApiException()
          .getObject();
      if (LOG.isDebugEnabled()) {
        LOG.debug("Get Notebook resource: \n{}", YamlUtils.toPrettyYaml(object));
      }
      notebook = NotebookUtils.parseObject(object, NotebookUtils.ParseOpt.PARSE_OPT_GET);
    } catch (ApiException e) {
      // SUBMARINE-1124
      // The exception that obtaining CRD resources is not necessarily because the CRD is deleted,
      // but maybe due to timeout or API error caused by network and other reasons.
      // Therefore, the status of the notebook should be set to a new enum NOTFOUND.
      LOG.warn("Get error when submitter is finding notebook: {}", getMetadata().getName());
      if (notebook == null) {
        notebook = new Notebook();
      }
      notebook.setReason(e.getMessage());
      notebook.setStatus(Notebook.Status.STATUS_NOT_FOUND.getValue());
    }
    return resetName(notebook);
  }
}
