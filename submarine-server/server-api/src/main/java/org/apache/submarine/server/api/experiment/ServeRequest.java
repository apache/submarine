package org.apache.submarine.server.api.experiment;

public class ServeRequest {
  // String modelName, String modelVersion, String namespace
  private String modelName;
  private String modelVersion;
  private String namespace;


  public ServeRequest() {
  }

  public ServeRequest(String modelName, String modelVersion, String namespace) {
    this.modelName = modelName;
    this.modelVersion = modelVersion;
    this.namespace = namespace;
  }

  public String getModelName() {
    return this.modelName;
  }

  public void setModelName(String modelName) {
    this.modelName = modelName;
  }

  public String getModelVersion() {
    return this.modelVersion;
  }

  public void setModelVersion(String modelVersion) {
    this.modelVersion = modelVersion;
  }

  public String getNamespace() {
    return this.namespace;
  }

  public void setNamespace(String namespace) {
    this.namespace = namespace;
  }

  public ServeRequest modelName(String modelName) {
    setModelName(modelName);
    return this;
  }

  public ServeRequest modelVersion(String modelVersion) {
    setModelVersion(modelVersion);
    return this;
  }

  public ServeRequest namespace(String namespace) {
    setNamespace(namespace);
    return this;
  }

  @Override
  public String toString() {
    return "{" +
      " modelName='" + getModelName() + "'" +
      ", modelVersion='" + getModelVersion() + "'" +
      ", namespace='" + getNamespace() + "'" +
      "}";
  }
}
