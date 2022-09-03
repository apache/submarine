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
package org.apache.submarine.serve.tensorflow;

import io.kubernetes.client.openapi.models.V1ObjectMetaBuilder;
import org.apache.submarine.serve.seldon.PredictorAnnotations;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonDeploymentSpec;
import org.apache.submarine.serve.seldon.SeldonGraph;
import org.apache.submarine.serve.seldon.SeldonPredictor;
import org.apache.submarine.serve.utils.SeldonConstants;
import org.apache.submarine.server.k8s.utils.K8sUtils;

public class SeldonTFServing extends SeldonDeployment {

  public SeldonTFServing() {
  }

  public SeldonTFServing(String resourceName, String modelName, String modelURI) {
    V1ObjectMetaBuilder metaBuilder = new V1ObjectMetaBuilder();
    metaBuilder.withNamespace(K8sUtils.getNamespace())
        .withName(resourceName)
        .addToLabels(MODEL_NAME_LABEL, modelName);
    setMetadata(metaBuilder.build());

    setSpec(new SeldonDeploymentSpec(SeldonConstants.SELDON_PROTOCOL));

    SeldonGraph seldonGraph = new SeldonGraph();
    seldonGraph.setName(modelName);
    seldonGraph.setImplementation(SeldonConstants.TFSERVING_IMPLEMENTATION);
    seldonGraph.setModelUri(modelURI);
    SeldonPredictor seldonPredictor = new SeldonPredictor();
    seldonPredictor.setAnnotations(PredictorAnnotations.service(resourceName));
    seldonPredictor.setSeldonGraph(seldonGraph);

    addPredictor(seldonPredictor);
  }
}
