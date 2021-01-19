package org.apache.submarine.server.experiment.database;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class ExperimentServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(ExperimentServiceTest.class);
  ExperimentService experimentService = new ExperimentService();

  @After
  public void cleanExperimentTable() throws Exception {
    List<ExperimentEntity> entities = experimentService.selectAll();
    for (ExperimentEntity entity: entities) {
      experimentService.delete(entity.getId());
    }
  }

  @Test
  public void testInsert() throws Exception {
    try {
      ExperimentEntity entity = new ExperimentEntity();
      String id = "experiment_1230";
      String spec = "{\"value\": 1}";

      entity.setId(id);
      entity.setExperimentSpec(spec);

      experimentService.insert(entity);

      ExperimentEntity entitySelected = experimentService.select(id);

      compareEntity(entity, entitySelected);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
  }

  @Test
  public void testSelectAll() throws Exception  {
    try {
      final int SIZE = 3;
      List<ExperimentEntity> entities = new ArrayList<ExperimentEntity>();

      for (int i = 0; i < SIZE; i++) {
        ExperimentEntity entity = new ExperimentEntity();
        entity.setId(String.format("experiment_%d", i));
        entity.setExperimentSpec(String.format("{\"value\": %d}", i));
        experimentService.insert(entity);
        entities.add(entity);
      }

      List<ExperimentEntity> entities_selected = experimentService.selectAll();

      Assert.assertEquals(SIZE, entities_selected.size());
      for (int i = 0; i < entities_selected.size(); i++) {
        compareEntity(entities.get(i), entities_selected.get(i));
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
  };

  @Test
  public void testUpdate() throws Exception {
    try {
      ExperimentEntity entity = new ExperimentEntity();
      String id = "experiment_1230";
      String spec = "{\"value\": 1}";
      entity.setId(id);
      entity.setExperimentSpec(spec);
      experimentService.insert(entity);

      String new_spec = "{\"value\": 2}";
      entity.setExperimentSpec(new_spec);
      experimentService.update(entity);

      ExperimentEntity entitySelected = experimentService.select(id);
      compareEntity(entity, entitySelected);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
  };

  @Test
  public void testDelete() throws Exception {
    try {
      ExperimentEntity entity = new ExperimentEntity();
      String id = "experiment_1230";
      String spec = "{\"value\": 1}";

      entity.setId(id);
      entity.setExperimentSpec(spec);

      experimentService.insert(entity);

      experimentService.delete(id);

      List<ExperimentEntity> entitySelected = experimentService.selectAll();

      Assert.assertEquals(0, entitySelected.size());
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
  };

  private void compareEntity(ExperimentEntity expected, ExperimentEntity actual) {
    Assert.assertEquals(expected.getId(), actual.getId());
    Assert.assertEquals(expected.getExperimentSpec(), actual.getExperimentSpec());
  }
}
