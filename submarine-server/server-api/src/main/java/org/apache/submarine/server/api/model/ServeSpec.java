package org.apache.submarine.server.api.model;

public class ServeSpec {
  private String modelName;
  private Integer modelVersion;
  private String serveType;
  private String modelURI;

  public String getModelName() {
    return modelName;
  }

  public void setModelName(String modelName) {
    this.modelName = modelName;
  }

  public Integer getModelVersion() {
    return modelVersion;
  }

  public void setModelVersion(Integer modelVersion) {
    this.modelVersion = modelVersion;
  }

  public String getServeType() {
    return serveType;
  }

  public void setServeType(String serveType) {
    this.serveType = serveType;
  }

  public String getModelURI() {
    return modelURI;
  }

  public void setModelURI(String modelURI) {
    this.modelURI = modelURI;
  }
}
