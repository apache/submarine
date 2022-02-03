package org.apache.submarine.server.submitter.k8s.model;

import com.google.gson.annotations.SerializedName;

import io.kubernetes.client.common.KubernetesListObject;
import io.kubernetes.client.openapi.models.V1ListMeta;

import java.util.List;

public class NotebookCRList implements KubernetesListObject{

  public static final String CRD_NOTEBOOK_VERSION_V1 = "v1alpha1";
  public static final String CRD_NOTEBOOK_GROUP_V1 = "kubeflow.org";
  public static final String CRD_APIVERSION_V1 = CRD_NOTEBOOK_GROUP_V1 + "/" + CRD_NOTEBOOK_VERSION_V1;
  public static final String CRD_NOTEBOOK_LIST_KIND_V1 = "NotebookList";

  public NotebookCRList() {
    setApiVersion(CRD_APIVERSION_V1);
    setKind(CRD_NOTEBOOK_LIST_KIND_V1);
  }
  
  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ListMeta metadata;
    
  @SerializedName("items")
  private List<NotebookCR> items;
  
  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  @Override
  public V1ListMeta getMetadata() {
    return metadata;
  }

  @Override
  public List<NotebookCR> getItems() {
    return items;
  }

  @Override
  public String getApiVersion() {
    return apiVersion;
  }

  @Override
  public String getKind() {
  
    return kind;
  }
}

