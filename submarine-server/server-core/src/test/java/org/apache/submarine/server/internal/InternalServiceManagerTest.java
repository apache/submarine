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

package org.apache.submarine.server.internal;

import org.apache.submarine.commons.runtime.exception.SubmarineException;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.notebook.NotebookId;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.experiment.database.entity.ExperimentEntity;
import org.apache.submarine.server.experiment.database.service.ExperimentService;
import org.apache.submarine.server.notebook.database.service.NotebookService;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.HashMap;
import java.util.Map;

public class InternalServiceManagerTest {
  private final Logger LOG = LoggerFactory.getLogger(InternalServiceManagerTest.class);
  private InternalServiceManager internalServiceManager;
  private NotebookService notebookService;
  private ExperimentService experimentService;
  
  @Before
  public void init() throws SubmarineException {
    notebookService = mock(NotebookService.class);
    experimentService = mock(ExperimentService.class);
    internalServiceManager = new InternalServiceManager(experimentService, notebookService);
  }
  
  @Test
  public void testUpdateNotebook() {
    Map<String, Object> updateObject = new HashMap<>();  
    updateObject.put("name", "test");  
    updateObject.put("notebookId", new NotebookId());
    updateObject.put("reason", "test");  
    updateObject.put("spec", new NotebookSpec());
    updateObject.put("status", "running");
    updateObject.put("uid", "mock-user");
    updateObject.put("url", "http://submarine.org");
    
    when(notebookService.update(any(Notebook.class))).thenReturn(true);
      
    assertEquals(internalServiceManager.updateCRStatus(CustomResourceType.Notebook,
            updateObject.get("notebookId").toString(), updateObject), true);
  }
  
  @Test
  public void testUpdateExperiment() {
    ExperimentEntity experimentEntity = new ExperimentEntity();
    experimentEntity.setId("test");
    experimentEntity.setExperimentSpec("");
    experimentEntity.setExperimentStatus("running");
    
    Map<String, Object> updateObject = new HashMap<>();  
    updateObject.put("id", "test");  
    updateObject.put("experimentSpec", "");
    updateObject.put("status", "running");  
    
    when(experimentService.update(any(ExperimentEntity.class))).thenReturn(true);
    
    assertEquals(internalServiceManager.updateCRStatus(CustomResourceType.TFJob,
        experimentEntity.getId(), updateObject), true);
  }
}
