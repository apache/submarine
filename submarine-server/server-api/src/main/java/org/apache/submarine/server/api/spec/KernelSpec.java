package org.apache.submarine.server.api.spec;

import java.util.Map;

/**
 * 
 * Kernel Spec
 */
public class KernelSpec {
  
  /**
   * Name of the kernel
   */
  private String name;
  
  /**
   * Name of the channel
   */
  private String channels;
  
  /**
   * List of kernel dependencies
   */
  private Map<String, String> dependencies;

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public String getChannels() {
    return channels;
  }

  public void setChannels(String channels) {
    this.channels = channels;
  }

  public Map<String, String> getDependencies() {
    return dependencies;
  }

  public void setDependencies(Map<String, String> dependencies) {
    this.dependencies = dependencies;
  }  
}
