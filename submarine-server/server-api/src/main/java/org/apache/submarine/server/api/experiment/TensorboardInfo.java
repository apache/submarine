package org.apache.submarine.server.api.experiment;

public class TensorboardInfo {
  public boolean available;
  public String url;

  public TensorboardInfo(boolean available, String url) {
    this.available = available;
    this.url = url;
  }

  public boolean isAvailable() {
    return available;
  }

  public void setAvailable(boolean available) {
    this.available = available;
  }

  public String getUrl() {
    return url;
  }

  public void setUrl(String url) {
    this.url = url;
  }

  @Override
  public String toString() {
    return "TensorboardInfo{" +
      "available=" + available +
      ", url='" + url + '\'' +
      '}';
  }
}
