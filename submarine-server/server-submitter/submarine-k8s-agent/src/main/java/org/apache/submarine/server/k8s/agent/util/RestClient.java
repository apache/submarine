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

package org.apache.submarine.server.k8s.agent.util;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.core.MediaType;

import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.rest.RestConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RestClient {
  private static final Logger LOG = LoggerFactory.getLogger(RestClient.class);
  
  private Client client = ClientBuilder.newClient();
  private final String API_SERVER_URL;
  public RestClient(String host, Integer port) {
    LOG.info("SERVER_HOST:" + host);
    LOG.info("SERVER_PORT:" + port);
    API_SERVER_URL = String.format("http://%s:%d/", host, port);
  }
  
  public void callStatusUpdate(CustomResourceType type, String resourceId, Object updateObject) {
      
      String uri = String.format("api/%s/%s/%s/%s", RestConstants.V1,
              RestConstants.INTERNAL, type.toString(), resourceId);
      LOG.info("Targeting uri:" + uri);
            
      client.target(API_SERVER_URL)
      .path(uri)      
      .request(MediaType.APPLICATION_JSON)
      .post(Entity.entity(updateObject, MediaType.APPLICATION_JSON), String.class);        
  }
  
}
