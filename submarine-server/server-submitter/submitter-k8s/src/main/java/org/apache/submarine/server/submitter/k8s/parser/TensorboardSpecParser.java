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

import io.kubernetes.client.custom.IntOrString;
import io.kubernetes.client.models.V1Container;
import io.kubernetes.client.models.V1ContainerPort;
import io.kubernetes.client.models.V1Deployment;
import io.kubernetes.client.models.V1DeploymentSpec;
import io.kubernetes.client.models.V1LabelSelector;
import io.kubernetes.client.models.V1ObjectMeta;
import io.kubernetes.client.models.V1PersistentVolumeClaimVolumeSource;
import io.kubernetes.client.models.V1PodSpec;
import io.kubernetes.client.models.V1PodTemplateSpec;
import io.kubernetes.client.models.V1Service;
import io.kubernetes.client.models.V1ServicePort;
import io.kubernetes.client.models.V1ServiceSpec;
import io.kubernetes.client.models.V1Volume;
import io.kubernetes.client.models.V1VolumeMount;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRoute;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRouteSpec;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.SpecRoute;
import org.apache.submarine.server.submitter.k8s.util.TensorboardUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class TensorboardSpecParser {
  public static V1Deployment parseDeployment(String name, String image, String routePath, String pvc) {

    V1Deployment deployment = new V1Deployment();

    V1ObjectMeta deploymentMetedata = new V1ObjectMeta();
    deploymentMetedata.setName(name);
    deployment.setMetadata(deploymentMetedata);
    V1DeploymentSpec deploymentSpec = new V1DeploymentSpec();
    deploymentSpec.setSelector(
        new V1LabelSelector().matchLabels(Collections.singletonMap("app", name)) // match the template
    );

    V1PodTemplateSpec deploymentTemplateSpec = new V1PodTemplateSpec();
    deploymentTemplateSpec.setMetadata(
        new V1ObjectMeta().labels(Collections.singletonMap("app", name)) // bind to replicaset and service
    );

    V1PodSpec deploymentTemplatePodSpec = new V1PodSpec();

    V1Container container = new V1Container();
    container.setName(name);
    container.setImage(image);
    container.setCommand(Arrays.asList(
        "tensorboard", "--logdir=/logs",
        String.format("--path_prefix=%s", routePath)
    ));
    container.setImagePullPolicy("IfNotPresent");
    container.addPortsItem(new V1ContainerPort().containerPort(TensorboardUtils.DEFAULT_TENSORBOARD_PORT));
    container.addVolumeMountsItem(new V1VolumeMount().mountPath("/logs").name("volume"));
    deploymentTemplatePodSpec.addContainersItem(container);

    V1Volume volume = new V1Volume().name("volume");
    volume.setPersistentVolumeClaim(
        new V1PersistentVolumeClaimVolumeSource().claimName(pvc)
    );
    deploymentTemplatePodSpec.addVolumesItem(volume);

    deploymentTemplateSpec.setSpec(deploymentTemplatePodSpec);

    deploymentSpec.setTemplate(deploymentTemplateSpec);

    deployment.setSpec(deploymentSpec);

    return deployment;
  }

  public static V1Service parseService(String svcName, String podName) {
    V1Service svc = new V1Service();
    svc.metadata(new V1ObjectMeta().name(svcName));

    V1ServiceSpec svcSpec = new V1ServiceSpec();
    svcSpec.setSelector(Collections.singletonMap("app", podName)); // bind to pod
    svcSpec.addPortsItem(new V1ServicePort().protocol("TCP").targetPort(
        new IntOrString(TensorboardUtils.DEFAULT_TENSORBOARD_PORT)).port(TensorboardUtils.SERVICE_PORT));
    svc.setSpec(svcSpec);
    return svc;
  }

  public static IngressRoute parseIngressRoute(String ingressName, String namespace,
                                               String routePath, String svcName) {

    IngressRoute ingressRoute = new IngressRoute();
    ingressRoute.setMetadata(
        new V1ObjectMeta().name(ingressName).namespace((namespace))
    );

    IngressRouteSpec ingressRouteSpec = new IngressRouteSpec();
    ingressRouteSpec.setEntryPoints(new HashSet<>(Collections.singletonList("web")));
    SpecRoute specRoute = new SpecRoute();
    specRoute.setKind("Rule");
    specRoute.setMatch(String.format("PathPrefix(`%s`)", routePath));

    Map<String, Object> service = new HashMap<String, Object>() {{
        put("kind", "Service");
        put("name", svcName);
        put("port", TensorboardUtils.SERVICE_PORT);
        put("namespace", namespace);
      }};

    specRoute.setServices(new HashSet<Map<String, Object>>() {{
        add(service);
      }});

    ingressRouteSpec.setRoutes(new HashSet<SpecRoute>() {{
        add(specRoute);
      }});

    ingressRoute.setSpec(ingressRouteSpec);

    return ingressRoute;
  }

}
