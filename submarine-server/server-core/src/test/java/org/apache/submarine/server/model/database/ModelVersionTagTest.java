package org.apache.submarine.server.model.database;

import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.entities.ModelVersionTagEntity;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.ModelVersionService;
import org.apache.submarine.server.model.database.service.ModelVersionTagService;
import org.apache.submarine.server.model.database.service.RegisteredModelService;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelVersionTagTest {
  private static final Logger LOG = LoggerFactory.getLogger(ModelVersionTagTest.class);
  RegisteredModelService registeredModelService = new RegisteredModelService();
  ModelVersionService modelVersionService = new ModelVersionService();
  ModelVersionTagService modelVersionTagService = new ModelVersionTagService();


  @After
  public void cleanAll() {
    registeredModelService.deleteAll();
  }

  @Test
  public void testInsertAndDelete() {
    String name = "InsertModelVersionTag";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelService.insert(registeredModelEntity);

    Integer version = 1;
    ModelVersionEntity modelVersionEntity = new ModelVersionEntity();
    modelVersionEntity.setName(name);
    modelVersionEntity.setVersion(version);
    modelVersionEntity.setSource("path/to/source");
    modelVersionEntity.setUserId("test");
    modelVersionEntity.setExperimentId("application_1234");
    modelVersionService.insert(modelVersionEntity);

    ModelVersionTagEntity modelVersionTagEntity = new ModelVersionTagEntity();
    modelVersionTagEntity.setName(name);
    modelVersionTagEntity.setVersion(version);
    modelVersionTagEntity.setTag("tag");
    modelVersionTagService.insert(modelVersionTagEntity);

    ModelVersionEntity modelVersionEntitySelected = modelVersionService.selectWithTag(name, version);
    Assert.assertEquals(modelVersionTagEntity.getTag(), modelVersionEntitySelected.getTags().get(0));

    modelVersionTagService.delete(modelVersionTagEntity);
    modelVersionEntitySelected = modelVersionService.selectWithTag(name, version);
    Assert.assertEquals(0, modelVersionEntitySelected.getTags().size());
  }
}
