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

package org.apache.submarine.server.submitter.k8s.seldon;

import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.submitter.k8s.model.seldon.SeldonDeploymentFactory;
import org.apache.submarine.server.submitter.k8s.model.seldon.SeldonDeploymentPytorchServing;
import org.apache.submarine.server.submitter.k8s.model.seldon.SeldonDeploymentTFServing;
import org.apache.submarine.server.submitter.k8s.model.seldon.SeldonResource;
import org.apache.submarine.server.submitter.k8s.util.YamlUtils;
import org.junit.Assert;
import org.junit.Test;

/**
 * Test Seldon Deployment Resource
 */
public class SeldonDeploymentResourceTest {

  @Test
  public void testSeldonDeploymentTFServingResourceToYaml() {
    ServeSpec serveSpec = new ServeSpec();
    serveSpec.setId(1L);
    serveSpec.setModelVersion(1);
    serveSpec.setModelName("test-model");
    serveSpec.setModelType("tensorflow");
    serveSpec.setModelId("5f0a43f251064fa8979660eddf04ede8");
    serveSpec.setModelURI("s3://submarine/registry/" +
        "test-model-1-5f0a43f251064fa8979660eddf04ede8/test-model");

    // to yaml
    SeldonResource seldonDeployment = SeldonDeploymentFactory.getSeldonDeployment(serveSpec);
    String yaml = YamlUtils.toPrettyYaml(seldonDeployment);
    System.out.println(yaml);

    // cast to object
    SeldonDeploymentTFServing sdtfs = YamlUtils.readValue(yaml, SeldonDeploymentTFServing.class);
    Assert.assertEquals("submarine-model-1-5f0a43f251064fa8979660eddf04ede8",
        sdtfs.getMetadata().getName());
    Assert.assertEquals(1, sdtfs.getSpec().getPredictors().size());
    Assert.assertEquals("seldon", sdtfs.getSpec().getProtocol());
    Assert.assertEquals("version-1", sdtfs.getSpec().getPredictors().get(0).getSeldonGraph().getName());
  }

  @Test
  public void testSeldonDeploymentPytorchServingResourceToYaml() {
    ServeSpec serveSpec = new ServeSpec();
    serveSpec.setId(2L);
    serveSpec.setModelVersion(5);
    serveSpec.setModelName("test-model");
    serveSpec.setModelType("pytorch");
    serveSpec.setModelId("5f0a43f251064fa8979660eddf04ede8");
    serveSpec.setModelURI("s3://submarine/registry/" +
            "test-model-1-5f0a43f251064fa8979660eddf04ede8/test-model");

    // to yaml
    SeldonResource seldonDeployment = SeldonDeploymentFactory.getSeldonDeployment(serveSpec);
    String yaml = YamlUtils.toPrettyYaml(seldonDeployment);
    System.out.println(yaml);

    // cast to object
    SeldonDeploymentPytorchServing sdpts = YamlUtils.readValue(yaml, SeldonDeploymentPytorchServing.class);
    Assert.assertEquals("submarine-model-2-5f0a43f251064fa8979660eddf04ede8",
            sdpts.getMetadata().getName());
    Assert.assertEquals("kfserving", sdpts.getSpec().getProtocol());
    Assert.assertEquals(1, sdpts.getSpec().getPredictors().size());
    Assert.assertEquals("version-5", sdpts.getSpec().getPredictors().get(0).getSeldonGraph().getName());
  }

}
