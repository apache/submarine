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
package org.apache.submarine.serve.seldon.pytorch;

import org.apache.submarine.serve.seldon.PredictorAnnotations;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonDeploymentSpec;
import org.apache.submarine.serve.seldon.SeldonGraph;
import org.apache.submarine.serve.seldon.SeldonPredictor;
import org.apache.submarine.serve.utils.SeldonConstants;

public class SeldonPytorchServing extends SeldonDeployment {

  public SeldonPytorchServing() {
  }

  public SeldonPytorchServing(String resourceName, String modelName, Integer modelVersion,
                              String modelId, String modelURI) {
    super(resourceName, modelName, modelVersion, modelId, modelURI);

    setSpec(new SeldonDeploymentSpec(SeldonConstants.KFSERVING_PROTOCOL));

    SeldonGraph seldonGraph = new SeldonGraph();
    seldonGraph.setName(String.format("version-%s", modelVersion));
    seldonGraph.setImplementation(SeldonConstants.TRITON_IMPLEMENTATION);
    seldonGraph.setModelUri(modelURI);
    SeldonPredictor seldonPredictor = new SeldonPredictor();
    seldonPredictor.setAnnotations(PredictorAnnotations.service(resourceName));
    seldonPredictor.setSeldonGraph(seldonGraph);

    addPredictor(seldonPredictor);
  }
}
