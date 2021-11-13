package org.apache.submarine.serve.seldon;

import com.google.gson.annotations.SerializedName;

public class SeldonPredictor {
  @SerializedName("name")
  private String name = "default";

  @SerializedName("replicas")
  private Integer replicas = 1;

  @SerializedName("graph")
  private SeldonGraph seldonGraph = new SeldonGraph();

  public SeldonPredictor(String name, Integer replicas, SeldonGraph graph) {
    setName(name);
    setReplicas(replicas);
    setSeldonGraph(graph);
  }

  public SeldonPredictor(){};

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public Integer getReplicas() {
    return replicas;
  }

  public void setReplicas(Integer replicas) {
    this.replicas = replicas;
  }

  public SeldonGraph getSeldonGraph() {
    return seldonGraph;
  }

  public void setSeldonGraph(SeldonGraph seldonGraph) {
    this.seldonGraph = seldonGraph;
  }
}
