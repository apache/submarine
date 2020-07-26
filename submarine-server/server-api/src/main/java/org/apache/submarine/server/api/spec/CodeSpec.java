package org.apache.submarine.server.api.spec;

public class CodeSpec {
  
  private String sync_mode;
  
  private String url;
  
  public String getSyncMode() {
    return sync_mode;
  }

  public void setSyncMode(String syncMode) {
    this.sync_mode = syncMode;
  }

  public String getUrl() {
    return url;
  }

  public void setUrl(String url) {
    this.url = url;
  }
}
