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

package org.apache.submarine.rest;

import com.google.gson.Gson;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.io.FileUtils;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.RestConstants;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.BeforeClass;

import javax.ws.rs.core.Response;
import java.io.File;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;

public class JobManagerRestApiIT extends AbstractSubmarineServerTest {
  private static final Logger LOG = LoggerFactory.getLogger(JobManagerRestApiIT.class);

  @BeforeClass
  public static void startUp(){
    Assert.assertTrue(checkIfServerIsRunning());
  }

  // Test job created with correct JSON input
  @Test
  public void testCreateJob() throws Exception {
    URL fileUrl = this.getClass().getResource("/tf-mnist-req.json");
    String jobSpec = FileUtils.readFileToString(new File(fileUrl.toURI()), StandardCharsets.UTF_8);

    PostMethod response = httpPost("/api/" + RestConstants.V1 + "/" + RestConstants.JOBS, jobSpec);
    LOG.debug(response.toString());

    String responseBodyAsString = response.getResponseBodyAsString();
    LOG.debug(responseBodyAsString);

    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(responseBodyAsString, JsonResponse.class);
    assertEquals("Response code should be 200 ",
        Response.Status.ACCEPTED.getStatusCode(), jsonResponse.getCode());
  }
}
