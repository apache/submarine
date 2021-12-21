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
package org.apache.submarine.serve.pytorch;

import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonGraph;
import org.apache.submarine.serve.seldon.SeldonPredictor;
import org.apache.submarine.serve.utils.SeldonConstants;

public class SeldonPytorchServing extends SeldonDeployment {
  public SeldonPytorchServing(String name, String modelURI){
    V1ObjectMeta v1ObjectMeta = new V1ObjectMeta();
    v1ObjectMeta.setName(name);
    v1ObjectMeta.setNamespace(SeldonConstants.DEFAULT_NAMESPACE);
    setMetadata(v1ObjectMeta);

    setSpec(new SeldonDeploymentSpec(SeldonConstants.KFSERVING_PROTOCOL));

    SeldonGraph seldonGraph = new SeldonGraph();
    seldonGraph.setName(name);
    seldonGraph.setImplementation(SeldonConstants.TRITON_IMPLEMENTATION);
    seldonGraph.setModelUri(modelURI);
    SeldonPredictor seldonPredictor = new SeldonPredictor();
    seldonPredictor.setSeldonGraph(seldonGraph);

    addPredictor(seldonPredictor);
  }
}
