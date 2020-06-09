package org.apache.submarine.server.api.spec;

import java.util.List;

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
   * Name of the channels
   */
  private List<String> channels;
  
  /**
   * List of kernel dependencies
   */
  private List<String> dependencies;

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public List<String> getChannels() {
    return channels;
  }

  public void setChannels(List<String> channels) {
    this.channels = channels;
  }

  public List<String> getDependencies() {
    return dependencies;
  }

  public void setDependencies(List<String> dependencies) {
    this.dependencies = dependencies;
  }  
}
