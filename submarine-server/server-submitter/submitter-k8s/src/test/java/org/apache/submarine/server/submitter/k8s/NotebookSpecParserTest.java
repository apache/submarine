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

import io.kubernetes.client.openapi.models.V1ConfigMap;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.server.api.spec.NotebookMeta;
import org.apache.submarine.server.api.spec.NotebookPodSpec;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.submitter.k8s.model.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.NotebookCRSpec;
import org.apache.submarine.server.submitter.k8s.parser.ConfigmapSpecParser;
import org.apache.submarine.server.submitter.k8s.parser.NotebookSpecParser;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;

public class NotebookSpecParserTest extends SpecBuilder {

  @Test
  public void testValidNotebook() throws IOException, URISyntaxException {
    NotebookSpec notebookSpec = (NotebookSpec) buildFromJsonFile(NotebookSpec.class, notebookReqFile);
    NotebookCR notebook = NotebookSpecParser.parseNotebook(notebookSpec);

    validateMetadata(notebookSpec.getMeta(), notebook.getMetadata());
    validateEnvironment(notebookSpec, notebook.getSpec());
    validatePodSpec(notebookSpec.getSpec(), notebook);
  }

  private void validateMetadata(NotebookMeta meta, V1ObjectMeta actualMeta) {
    Assert.assertEquals(meta.getName(), actualMeta.getName());
    Assert.assertEquals(meta.getNamespace(), actualMeta.getNamespace());
    Assert.assertEquals(meta.getOwnerId(),
            actualMeta.getLabels().get(NotebookCR.NOTEBOOK_OWNER_SELECTOR_KEY));
  }

  private void validateEnvironment(NotebookSpec spec, NotebookCRSpec actualPodSpec) {
    String expectedImage = spec.getEnvironment().getImage();
    String actualImage = actualPodSpec.getContainerImageName();
    Assert.assertEquals(expectedImage, actualImage);
  }

  private void validatePodSpec(NotebookPodSpec podSpec, NotebookCR notebook) {
    NotebookCRSpec notebookCRSpec = null;
    if (notebook != null) {
      notebookCRSpec = notebook.getSpec();
    }
    Assert.assertNotNull(notebookCRSpec);

    // environment variable
    for (Map.Entry<String, String> entry : podSpec.getEnvVars().entrySet()) {
      V1EnvVar env = new V1EnvVar();
      env.setName(entry.getKey());
      env.setValue(env.getValue());
      Assert.assertTrue(notebook.getSpec().getEnvs().contains(env));
    }

    // mem
    String expectedContainerMem = podSpec.getMemory();
    String actualContainerMem = notebookCRSpec.getContainerMemory();
    Assert.assertEquals(expectedContainerMem, actualContainerMem);

    // cpu
    String expectedContainerCpu = podSpec.getCpu();
    String actualContainerCpu = notebookCRSpec.getContainerCpu();
    Assert.assertEquals(expectedContainerCpu, actualContainerCpu);
  }

  @Test
  public void testConfigMap() {
    String overwriteJson = "{ \"@jupyterlab/translation-extension:plugin\": " +
            "{ \"locale\": \"zh_CN\" } }";
    V1ConfigMap configMap = ConfigmapSpecParser.parseConfigMap("test",
            "overwrite.json", overwriteJson);
    Map<String, String> data = configMap.getData();
    Assert.assertEquals(data.size(), 1);
    Assert.assertEquals(data.get("overwrite.json"), overwriteJson);

    V1ConfigMap configMap2 = ConfigmapSpecParser.parseConfigMap("test", data);
    Map<String, String> data2 = configMap2.getData();
    Assert.assertEquals(data2.size(), 1);
    Assert.assertEquals(data2.get("overwrite.json"), overwriteJson);
  }
}
