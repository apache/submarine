package org.apache.submarine.server.model.database;

import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.entities.RegisteredModelTagEntity;
import org.apache.submarine.server.model.database.service.RegisteredModelService;
import org.apache.submarine.server.model.database.service.RegisteredModelTagService;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegisteredModelTagServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(RegisteredModelTagServiceTest.class);
  RegisteredModelService registeredModelService = new RegisteredModelService();
  RegisteredModelTagService registeredModelTagService = new RegisteredModelTagService();

  @After
  public void cleanAll() throws Exception {
    registeredModelService.deleteAll();
  }

  @Test
  public void testInsertAndDelete() {
    String name = "InsertRegisteredModelTag";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelService.insert(registeredModelEntity);

    RegisteredModelTagEntity registeredModelTagEntity = new RegisteredModelTagEntity();
    registeredModelTagEntity.setName(name);
    registeredModelTagEntity.setTag("tag");
    registeredModelTagService.insert(registeredModelTagEntity);

    RegisteredModelEntity registeredModelEntitySelected = registeredModelService.selectWithTag(name);
    Assert.assertEquals(registeredModelTagEntity.getTag(), registeredModelEntitySelected.getTags().get(0));

    registeredModelTagService.delete(registeredModelTagEntity);
    registeredModelEntitySelected = registeredModelService.selectWithTag(name);
    Assert.assertEquals(0, registeredModelEntitySelected.getTags().size());
  }
}
