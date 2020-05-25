package org.apache.submarine.server.api.environment;

import org.apache.submarine.server.api.spec.EnvironmentSpec;

public class Environment {

  /**
   * ID of the environment
   */
  private int environmentId;
  
  /**
   * Name of the environment
   */
  private String name;
  
  /**
   * Environment Spec
   */
  private EnvironmentSpec environmentSpec;

  public int getEnvironmentId() {
    return environmentId;
  }

  public void setEnvironmentId(int environmentId) {
    this.environmentId = environmentId;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public EnvironmentSpec getEnvironmentSpec() {
    return environmentSpec;
  }

  public void setEnvironmentSpec(EnvironmentSpec environmentSpec) {
    this.environmentSpec = environmentSpec;
  }
}
