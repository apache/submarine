package org.apache.submarine.server.submitter.k8s.model.tfjob;

import com.google.gson.annotations.SerializedName;
import java.util.List;

import io.kubernetes.client.common.KubernetesListObject;
import io.kubernetes.client.openapi.models.V1ListMeta;

public class TFJobList implements KubernetesListObject{

  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ListMeta metadata;

  @SerializedName("items")
  private List<TFJob> items;

  @Override
  public V1ListMeta getMetadata() {
    return metadata;
  }

  @Override
  public List<TFJob> getItems() {
    return items;
  }

  @Override
  public String getApiVersion() {
    return TFJob.CRD_TF_API_VERSION_V1;
  }

  @Override
  public String getKind() {
    return TFJob.CRD_TF_KIND_V1 + "List";
  }
}
