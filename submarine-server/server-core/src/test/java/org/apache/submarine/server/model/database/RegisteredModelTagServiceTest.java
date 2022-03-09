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

import org.apache.submarine.server.database.model.entities.RegisteredModelEntity;
import org.apache.submarine.server.database.model.entities.RegisteredModelTagEntity;
import org.apache.submarine.server.database.model.service.RegisteredModelService;
import org.apache.submarine.server.database.model.service.RegisteredModelTagService;
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
  public void cleanAll() {
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
