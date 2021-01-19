package org.apache.submarine.server.experiment.database;

import java.util.List;

public interface ExperimentMapper {
  List<ExperimentEntity> selectAll();
  ExperimentEntity select(String id);

  int insert(ExperimentEntity experiment);
  int update(ExperimentEntity experiment);
  int delete(String id);
}
