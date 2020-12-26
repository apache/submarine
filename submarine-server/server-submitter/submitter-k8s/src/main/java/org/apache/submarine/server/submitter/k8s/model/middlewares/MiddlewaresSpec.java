package org.apache.submarine.server.submitter.k8s.model.middlewares;

import com.google.gson.annotations.SerializedName;

import java.util.Map;

public class MiddlewaresSpec {

  @SerializedName("replacePathRegex")
  private Map<String, String> replacePathRegex;

  public MiddlewaresSpec() {
  }

  public Map<String, String> getReplacePathRegex() {
    return replacePathRegex;
  }

  public void setReplacePathRegex(Map<String, String> replacePathRegex) {
    this.replacePathRegex = replacePathRegex;
  }
}
