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

package org.apache.submarine.server.submitter.k8s.parser;

import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRoute;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRouteSpec;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.SpecRoute;
import org.apache.submarine.server.submitter.k8s.model.middlewares.Middlewares;
import org.apache.submarine.server.submitter.k8s.model.middlewares.MiddlewaresSpec;
import org.apache.submarine.server.submitter.k8s.model.middlewares.StripPrefix;

import io.kubernetes.client.custom.IntOrString;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1ContainerPort;
import io.kubernetes.client.openapi.models.V1Deployment;
import io.kubernetes.client.openapi.models.V1DeploymentSpec;
import io.kubernetes.client.openapi.models.V1HTTPGetAction;
import io.kubernetes.client.openapi.models.V1LabelSelector;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;
import io.kubernetes.client.openapi.models.V1Probe;
import io.kubernetes.client.openapi.models.V1Service;
import io.kubernetes.client.openapi.models.V1ServicePort;
import io.kubernetes.client.openapi.models.V1ServiceSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class ServeSpecParser {

  // names
  String generalName;
  String podName;
  String containerName;
  String routeName;
  String svcName;
  String middlewareName;

  // path
  String routePath;

  // model_uri
  String modelURI;

  // cluster related
  String namespace;
  int PORT = 5000; // mlflow serve server is listening on 5000

  // constructor
  public ServeSpecParser(String modelName, String modelVersion, String namespace) {
    // names assignment
    generalName = modelName + "-" + modelVersion;
    podName = generalName + "-pod";
    containerName = generalName + "-container";
    routeName = generalName + "-ingressroute";
    svcName = generalName + "-service";
    middlewareName = generalName + "-middleware";
    // path assignment
    routePath = String.format("/serve/%s", generalName);
    // uri assignment
    modelURI = String.format("models:/%s/%s", modelName, modelVersion);
    // nameSpace
    this.namespace = namespace;
  }

  public V1Deployment getDeployment() {
    // Container related
    // TODO(byronhsu) This should not be hard-coded.
    final String serveImage =
        "apache/submarine:serve-0.7.0-SNAPSHOT";

    ArrayList<String> cmds = new ArrayList<>(
        Arrays.asList("mlflow", "models", "serve",
        "--model-uri", modelURI, "--host", "0.0.0.0")
    );

    V1Deployment deployment = new V1Deployment();

    V1ObjectMeta deploymentMetedata = new V1ObjectMeta();
    deploymentMetedata.setName(generalName);
    deployment.setMetadata(deploymentMetedata);

    V1DeploymentSpec deploymentSpec = new V1DeploymentSpec();
    deploymentSpec.setSelector(
        new V1LabelSelector().matchLabels(Collections.singletonMap("app", podName)) // match the template
    );

    V1PodTemplateSpec deploymentTemplateSpec = new V1PodTemplateSpec();
    deploymentTemplateSpec.setMetadata(
        new V1ObjectMeta().labels(Collections.singletonMap("app", podName)) // bind to replicaset and service
    );

    V1PodSpec deploymentTemplatePodSpec = new V1PodSpec();

    V1Container container = new V1Container();
    container.setName(containerName);
    container.setImage(serveImage);
    container.setCommand(cmds);
    container.setImagePullPolicy("IfNotPresent");
    container.addPortsItem(new V1ContainerPort().containerPort(PORT));
    container.setReadinessProbe(
        new V1Probe().httpGet(new V1HTTPGetAction().path("/ping").port(new IntOrString(PORT)))
    );


    deploymentTemplatePodSpec.addContainersItem(container);
    deploymentTemplateSpec.setSpec(deploymentTemplatePodSpec);
    deploymentSpec.setTemplate(deploymentTemplateSpec);
    deployment.setSpec(deploymentSpec);

    return deployment;
  }
  public V1Service getService() {
    V1Service svc = new V1Service();
    svc.metadata(new V1ObjectMeta().name(svcName));

    V1ServiceSpec svcSpec = new V1ServiceSpec();
    svcSpec.setSelector(Collections.singletonMap("app", podName)); // bind to pod
    svcSpec.addPortsItem(new V1ServicePort().protocol("TCP").targetPort(
        new IntOrString(PORT)).port(PORT));
    svc.setSpec(svcSpec);
    return svc;
  }

  public IngressRoute getIngressRoute() {
    IngressRoute ingressRoute = new IngressRoute();
    ingressRoute.setMetadata(
        new V1ObjectMeta().name(routeName).namespace((namespace))
    );

    IngressRouteSpec ingressRouteSpec = new IngressRouteSpec();
    ingressRouteSpec.setEntryPoints(new HashSet<>(Collections.singletonList("web")));
    SpecRoute specRoute = new SpecRoute();
    specRoute.setKind("Rule");
    specRoute.setMatch(String.format("PathPrefix(`%s`)", routePath));

    Map<String, Object> service = new HashMap<String, Object>() {{
        put("kind", "Service");
        put("name", svcName);
        put("port", PORT);
        put("namespace", namespace);
      }};

    specRoute.setServices(new HashSet<Map<String, Object>>() {{
        add(service);
      }});

    Map<String, String> middleware = new HashMap<String, String>() {{
        put("name", middlewareName);
      }};

    specRoute.setMiddlewares(new HashSet<Map<String, String>>() {{
        add(middleware);
      }});

    ingressRouteSpec.setRoutes(new HashSet<SpecRoute>() {{
        add(specRoute);
      }});

    ingressRoute.setSpec(ingressRouteSpec);
    return ingressRoute;
  }

  public Middlewares getMiddlewares() {
    Middlewares middleware = new Middlewares();
    middleware.setMetadata(new V1ObjectMeta().name(middlewareName).namespace(namespace));

    MiddlewaresSpec middlewareSpec = new MiddlewaresSpec().stripPrefix(
        new StripPrefix().prefixes(Arrays.asList(routePath))
    );
    middleware.setSpec(middlewareSpec);
    return middleware;
  }


  public String getGeneralName() {
    return this.generalName;
  }

  public String getPodName() {
    return this.podName;
  }

  public String getContainerName() {
    return this.containerName;
  }
  public String getRouteName() {
    return this.routeName;
  }

  public String getSvcName() {
    return this.svcName;
  }

  public String getMiddlewareName() {
    return this.middlewareName;
  }

  public String getRoutePath() {
    return this.routePath;
  }

  public String getModelURI() {
    return this.modelURI;
  }

  public String getNamespace() {
    return this.namespace;
  }
}
