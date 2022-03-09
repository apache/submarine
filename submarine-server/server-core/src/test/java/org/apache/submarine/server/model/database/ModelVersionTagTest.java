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

package org.apache.submarine.server.database.model;

import org.apache.submarine.server.database.model.entities.ModelVersionEntity;
import org.apache.submarine.server.database.model.entities.ModelVersionTagEntity;
import org.apache.submarine.server.database.model.entities.RegisteredModelEntity;
import org.apache.submarine.server.database.model.service.ModelVersionService;
import org.apache.submarine.server.database.model.service.ModelVersionTagService;
import org.apache.submarine.server.database.model.service.RegisteredModelService;
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
    modelVersionEntity.setId("model_version_id");
    modelVersionEntity.setUserId("test");
    modelVersionEntity.setExperimentId("application_1234");
    modelVersionEntity.setModelType("tensorflow");
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
