/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.rest;

import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.submarine.server.AbstractSubmarineServerTest;

import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.BeforeClass;
import org.junit.Test;

public class ClusterRestApiIT extends AbstractSubmarineServerTest {
  public final static Logger LOG = LoggerFactory.getLogger(ClusterRestApiIT.class);

  @BeforeClass
  public static void startUp(){
    Assert.assertTrue(checkIfServerIsRunning());
  }

  @Test
  public void getClusterAddress() throws Exception {
    GetMethod get = httpGet("/api/v1/cluster/address");
    Assert.assertEquals(200, get.getStatusCode());
    String body = get.getResponseBodyAsString();
    LOG.info("body = {}", body);
  }

  @Test
  public void getClusterNodes() throws Exception {
    GetMethod get = httpGet("/api/v1/cluster/nodes");
    Assert.assertEquals(200, get.getStatusCode());
    String body = get.getResponseBodyAsString();
    LOG.info("body = {}", body);
  }
}
