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

package org.apache.submarine.serve.seldon;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.junit.Assert;
import org.junit.Test;

public class SeldonDeploymentTest {

  private static final Gson gson = new Gson();

  @Test
  public void testPredictorAnnotations() {
    PredictorAnnotations annotations = PredictorAnnotations
            .service("submarine-model-1-c9d95a3881b941148b0e2a6362605c00");
    JsonObject json = gson.toJsonTree(annotations).getAsJsonObject();
    Assert.assertEquals(json.size(), 2);
    Assert.assertEquals(
            json.get("seldon.io/svc-name").getAsString(),
            "submarine-model-1-c9d95a3881b941148b0e2a6362605c00"
    );
    Assert.assertEquals(
            json.get("traffic.sidecar.istio.io/excludeOutboundPorts").getAsString(),
            "9000"
    );
  }

}
