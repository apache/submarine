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

package org.apache.submarine.server.submitter.k8s;

import com.github.tomakehurst.wiremock.junit.WireMockRule;
import com.github.tomakehurst.wiremock.matching.EqualToPattern;
import io.kubernetes.client.openapi.models.V1ConfigMap;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.util.generic.KubernetesApiResponse;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.client.K8sMockClient;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.io.IOException;
import java.util.Collections;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class SubmitterK8sMockApiTest {

  private K8sClient k8sClient;

  @Rule
  public WireMockRule wireMockRule = K8sMockClient.getWireMockRule();

  @Before
  public void setup() throws IOException {
    this.k8sClient = new K8sMockClient(post(urlPathEqualTo("/api/v1/namespaces/foo/configmaps"))
            .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
            .willReturn(
                    aResponse()
                            .withStatus(200)
                            .withBody("{\"metadata\":{\"name\":\"bar\",\"namespace\":\"foo\"}}")));
  }

  @Test
  public void testApplyConfigMap() {
    V1ConfigMap configMap = new V1ConfigMap()
            .apiVersion("v1")
            .metadata(new V1ObjectMeta().namespace("foo").name("bar"))
            .data(Collections.singletonMap("key1", "value1"));

    KubernetesApiResponse<V1ConfigMap> configMapResp = k8sClient.getConfigMapClient().create(configMap);
    V1ConfigMap rtnConfigmap = configMapResp.getObject();

    assertNotNull(rtnConfigmap);
    assertEquals(rtnConfigmap.getMetadata().getNamespace(), "foo");
    assertEquals(rtnConfigmap.getMetadata().getName(), "bar");
  }

}
