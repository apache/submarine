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

import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.RegisteredModelService;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class RegisteredModelServiceTest {
  RegisteredModelService registeredModelService = new RegisteredModelService();

  @After
  public void cleanAll() {
    registeredModelService.deleteAll();
  }

  @Test
  public void testSelectAll() {
    String name = "RegisteredModel1";
    List<String> tags = new ArrayList<>();
    tags.add("tag1");
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelEntity.setTags(tags);
    registeredModelService.insert(registeredModelEntity);

    String name2 = "RegisteredModel2";
    List<String> tags2 = new ArrayList<>();
    tags2.add("tag2");
    RegisteredModelEntity registeredModelEntity2 = new RegisteredModelEntity();
    registeredModelEntity2.setName(name2);
    registeredModelEntity2.setTags(tags2);
    registeredModelService.insert(registeredModelEntity2);

    List<RegisteredModelEntity> registeredModelEntities = registeredModelService.selectAll();
    compareRegisteredModel(registeredModelEntity, registeredModelEntities.get(0));
    compareTags(registeredModelEntity, registeredModelEntities.get(0));
    compareRegisteredModel(registeredModelEntity2, registeredModelEntities.get(1));
    compareTags(registeredModelEntity2, registeredModelEntities.get(1));
  }

  @Test
  public void testInsertAndSelect() {
    String name = "RegisteredModel";
    String description = "Description.";
    List<String> tags = new ArrayList<>();
    tags.add("tag");
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelEntity.setDescription(description);
    registeredModelEntity.setTags(tags);
    registeredModelService.insert(registeredModelEntity);

    RegisteredModelEntity registeredModelEntitySelected = registeredModelService.select(name);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelected);

    RegisteredModelEntity registeredModelEntitySelectedWithTags = registeredModelService.selectWithTag(name);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelectedWithTags);
    compareTags(registeredModelEntity, registeredModelEntitySelectedWithTags);
  }

  @Test
  public void testUpdate() {
    String name = "RegisteredModel";
    String description = "Description.";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelEntity.setDescription(description);
    registeredModelService.insert(registeredModelEntity);

    RegisteredModelEntity registeredModelEntitySelected = registeredModelService.select(name);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelected);

    String newDescription = "New description.";
    registeredModelEntity.setDescription(newDescription);
    registeredModelService.update(registeredModelEntity);

    registeredModelEntitySelected = registeredModelService.select(name);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelected);
  }

  @Test
  public void testRename() {
    String name = "RegisteredModel";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelService.insert(registeredModelEntity);

    RegisteredModelEntity registeredModelEntitySelected = registeredModelService.select(name);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelected);

    String newName = "newRegisteredModel";
    registeredModelService.rename(name, newName);

    registeredModelEntitySelected = registeredModelService.select(newName);
    registeredModelEntity.setName(newName);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelected);
  }

  @Test
  public void testDelete() {
    String name = "RegisteredModel";
    RegisteredModelEntity registeredModelEntity = new RegisteredModelEntity();
    registeredModelEntity.setName(name);
    registeredModelService.insert(registeredModelEntity);

    RegisteredModelEntity registeredModelEntitySelected = registeredModelService.select(name);
    compareRegisteredModel(registeredModelEntity, registeredModelEntitySelected);

    registeredModelService.delete(name);
    List<RegisteredModelEntity> registeredModelEntities = registeredModelService.selectAll();

    Assert.assertEquals(0, registeredModelEntities.size());
  }

  private void compareRegisteredModel(RegisteredModelEntity expected, RegisteredModelEntity actual) {
    Assert.assertEquals(expected.getName(), actual.getName());
    Assert.assertNotNull(actual.getCreationTime());
    Assert.assertNotNull(actual.getLastUpdatedTime());
    Assert.assertEquals(expected.getDescription(), actual.getDescription());
  }

  private void compareTags(RegisteredModelEntity expected, RegisteredModelEntity actual) {
    Assert.assertEquals(true, expected.getTags().equals(actual.getTags()));
  }
}
