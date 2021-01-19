package org.apache.submarine.server.experiment.database;

import org.apache.submarine.server.database.entity.BaseEntity;

public class ExperimentEntity extends BaseEntity {
  /*
    Take id (inherited from BaseEntity) as the primary key for experiment table
  */
  private String experimentSpec;

  public String getExperimentSpec() {
    return experimentSpec;
  }

  public void setExperimentSpec(String experimentSpec) {
    this.experimentSpec = experimentSpec;
  }
}
