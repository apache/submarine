package org.apache.submarine.server.submitter.k8s.model.pytorchjob;

import com.google.gson.annotations.SerializedName;
import java.util.List;

import io.kubernetes.client.common.KubernetesListObject;
import io.kubernetes.client.common.KubernetesObject;
import io.kubernetes.client.openapi.models.V1ListMeta;

public class PyTorchJobList implements KubernetesListObject{

  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ListMeta metadata;

  @SerializedName("items")
  private List<PyTorchJob> items;

  @Override
  public V1ListMeta getMetadata() {
    return metadata;
  }

  @Override
  public List<? extends KubernetesObject> getItems() {
    return items;
  }

  @Override
  public String getApiVersion() {
    return PyTorchJob.CRD_PYTORCH_API_VERSION_V1;
  }

  @Override
  public String getKind() {
    return PyTorchJob.CRD_PYTORCH_KIND_V1 + "List";
  }
}
