/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.submarine.server.model.database;

import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.ModelVersionService;
import org.apache.submarine.server.model.database.service.RegisteredModelService;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class ModelVersionTest {
  RegisteredModelService registeredModelService = new RegisteredModelService();
  ModelVersionService modelVersionService = new ModelVersionService();

  @After
  public void cleanAll() {
    registeredModelService.deleteAll();
  }

  @Test
  public void testSelectAllVersions() {
    String name = "selectAllModelVersions";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelService.insert(registeredModelEntity);

    Integer version = 1;
    List<String> tags = new ArrayList<>();
    tags.add("tag");
    ModelVersionEntity modelVersionEntity = new ModelVersionEntity();
    modelVersionEntity.setName(name);
    modelVersionEntity.setVersion(version);
    modelVersionEntity.setSource("path/to/source");
    modelVersionEntity.setUserId("test");
    modelVersionEntity.setExperimentId("application_1234");
    modelVersionEntity.setTags(tags);
    modelVersionService.insert(modelVersionEntity);

    Integer version2 = 2;
    List<String> tags2 = new ArrayList<>();
    tags2.add("tag2");
    ModelVersionEntity modelVersionEntity2 = new ModelVersionEntity();
    modelVersionEntity2.setName(name);
    modelVersionEntity2.setVersion(version2);
    modelVersionEntity2.setSource("path/to/source2");
    modelVersionEntity2.setUserId("test");
    modelVersionEntity2.setExperimentId("application_1234");
    modelVersionEntity2.setTags(tags2);
    modelVersionService.insert(modelVersionEntity2);

    List<ModelVersionEntity> modelVersionEntities = modelVersionService.selectAllVersions(name);
    compareModelVersion(modelVersionEntity, modelVersionEntities.get(0));
    compareTags(modelVersionEntity, modelVersionEntities.get(0));
    compareModelVersion(modelVersionEntity2, modelVersionEntities.get(1));
    compareTags(modelVersionEntity2, modelVersionEntities.get(1));
  }

  @Test
  public void testInsertAndSelect() {
    String name = "insertModelVersion";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelService.insert(registeredModelEntity);

    Integer version = 1;
    List<String> tags = new ArrayList<>();
    tags.add("tag");
    ModelVersionEntity modelVersionEntity = new ModelVersionEntity();
    modelVersionEntity.setName(name);
    modelVersionEntity.setVersion(version);
    modelVersionEntity.setSource("path/to/source");
    modelVersionEntity.setUserId("test");
    modelVersionEntity.setExperimentId("application_1234");
    modelVersionEntity.setTags(tags);
    modelVersionService.insert(modelVersionEntity);

    ModelVersionEntity modelVersionEntitySelected = modelVersionService.select(name, version);
    this.compareModelVersion(modelVersionEntity, modelVersionEntitySelected);

    ModelVersionEntity modelVersionEntitySelectedWithTag = modelVersionService.selectWithTag(name, version);
    this.compareModelVersion(modelVersionEntity, modelVersionEntitySelectedWithTag);
    this.compareTags(modelVersionEntity, modelVersionEntitySelectedWithTag);
  }

  @Test
  public void testUpdate() {
    String name = "updateModelVersion";
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

    ModelVersionEntity modelVersionEntitySelected = modelVersionService.select(name, version);
    this.compareModelVersion(modelVersionEntity, modelVersionEntitySelected);

    String newStage = "Developing";
    String newDataset = "mnist";
    String newDescription = "New description.";
    modelVersionEntity.setCurrentStage(newStage);
    modelVersionEntity.setDataset(newDataset);
    modelVersionEntity.setDescription(newDescription);
    modelVersionService.update(modelVersionEntity);

    modelVersionEntitySelected = modelVersionService.select(name, version);
    this.compareModelVersion(modelVersionEntity, modelVersionEntitySelected);
  }

  @Test
  public void testDelete() {
    String name = "deleteModelVersion";
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

    modelVersionService.delete(name, version);

  }

  private void compareModelVersion(ModelVersionEntity expected, ModelVersionEntity actual) {
    Assert.assertEquals(expected.getName(), actual.getName());
    Assert.assertEquals(expected.getVersion(), actual.getVersion());
    Assert.assertEquals(expected.getSource(), actual.getSource());
    Assert.assertEquals(expected.getUserId(), actual.getUserId());
    Assert.assertEquals(expected.getExperimentId(), actual.getExperimentId());
    Assert.assertEquals(expected.getCurrentStage(), actual.getCurrentStage());
    Assert.assertNotNull(actual.getCreationTime());
    Assert.assertNotNull(actual.getLastUpdatedTime());
    Assert.assertEquals(expected.getDataset(), actual.getDataset());
    Assert.assertEquals(expected.getDescription(), actual.getDescription());
  }

  private void compareTags(ModelVersionEntity expected, ModelVersionEntity actual) {
    Assert.assertEquals(expected.getTags(), actual.getTags());
  }
}
