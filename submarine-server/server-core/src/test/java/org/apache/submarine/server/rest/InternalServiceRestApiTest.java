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

package org.apache.submarine.server.rest;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import javax.ws.rs.core.Response;

import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.internal.InternalServiceManager;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.Before;
import org.junit.Test;


public class InternalServiceRestApiTest {
  InternalServiceRestApi internalServiceRestApi;

  @Before
  public void init() {
    InternalServiceManager internalServiceManager = mock(InternalServiceManager.class);
    internalServiceRestApi = mock(InternalServiceRestApi.class);
    internalServiceRestApi.setInternalServiceManager(internalServiceManager);
  }

  @Test
  public void testUpdateCRStatus() {
    when(internalServiceRestApi.updateEnvironment(any(String.class),
        any(String.class), any(String.class))).thenReturn(new JsonResponse.
        Builder<String>(Response.Status.OK).
        success(true).build());

    Response response = internalServiceRestApi.updateEnvironment(CustomResourceType.
            Notebook.getCustomResourceType(), "notebookId", "running");
    assertEquals(response.getStatus(), Response.Status.OK.getStatusCode());
  }
}
