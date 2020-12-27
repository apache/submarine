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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class TensorboardSpecParser {
  public static V1Deployment parseDeployment(String name, String image, String route_path, String pvc) {

    /*
    [start] Deployment
     */
    V1Deployment deployment = new V1Deployment();

    // [start] deployment - metadata
    V1ObjectMeta deployment_metedata = new V1ObjectMeta();
    deployment_metedata.setName(name);
    deployment.setMetadata(deployment_metedata);
    // [end] deployment - metadata

    // [start] deployment - spec
    V1DeploymentSpec deployment_spec = new V1DeploymentSpec();
    deployment_spec.setSelector(
        new V1LabelSelector().matchLabels(Collections.singletonMap("app", name)) // match the template
    );

    // [start] deployment - spec - template
    V1PodTemplateSpec deployment_template_spec = new V1PodTemplateSpec();
    deployment_template_spec.setMetadata(
        new V1ObjectMeta().labels(Collections.singletonMap("app", name)) // bind to replicaset and service
    );

    // [start] deployment - spec - template - podspec
    V1PodSpec deployment_template_pod_spec = new V1PodSpec();

    // [start] deployment - spec - template - podspec - container
    V1Container container = new V1Container();
    container.setName(name);
    container.setImage(image);
    container.setCommand(Arrays.asList(
        "tensorboard", "--logdir=/logs",
        String.format("--path_prefix=%s", route_path)
    ));
    container.setImagePullPolicy("IfNotPresent");
    container.addPortsItem(new V1ContainerPort().containerPort(6006));
    container.addVolumeMountsItem(new V1VolumeMount().mountPath("/logs").name("volume"));
    deployment_template_pod_spec.addContainersItem(container);
    // [end] deployment - spec - template - podspec - container

    // [start] deployment - spec - template - podspec - volume
    V1Volume volume = new V1Volume().name("volume");
    volume.setPersistentVolumeClaim(
        new V1PersistentVolumeClaimVolumeSource().claimName(pvc)
    );
    deployment_template_pod_spec.addVolumesItem(volume);
    // [end] deployment - spec - template - podspec - volume

    deployment_template_spec.setSpec(deployment_template_pod_spec);
    // [end] deployment - spec - template - podspec

    deployment_spec.setTemplate(deployment_template_spec);
    // [end] deployment - spec - template

    deployment.setSpec(deployment_spec);
    // [end] deployment - spec
    return deployment;
  }

  public static V1Service parseService(String svc_name, String pod_name) {
    V1Service svc = new V1Service();
    svc.metadata(new V1ObjectMeta().name(svc_name));

    V1ServiceSpec svc_spec = new V1ServiceSpec();
    svc_spec.setSelector(Collections.singletonMap("app", pod_name)); // bind to pod
    svc_spec.addPortsItem(new V1ServicePort().protocol("TCP").targetPort(new IntOrString(6006)).port(8080));
    svc.setSpec(svc_spec);
    return svc;
  }

  public static IngressRoute parseIngressRoute(String ingress_name, String namespace,
                                               String route_path, String svc_name) {

    IngressRoute ingressRoute = new IngressRoute();
    ingressRoute.setMetadata(
        new V1ObjectMeta().name(ingress_name).namespace((namespace))
    );

    IngressRouteSpec ingressRoute_spec = new IngressRouteSpec();
    ingressRoute_spec.setEntryPoints(new HashSet<>(Collections.singletonList("web")));
    SpecRoute spec_route = new SpecRoute();
    spec_route.setKind("Rule");
    spec_route.setMatch(String.format("PathPrefix(`%s`)", route_path));

    Map<String, Object> service = new HashMap<String, Object>() {{
        put("kind", "Service");
        put("name", svc_name);
        put("port", 8080);
        put("namespace", namespace);
      }};

    spec_route.setServices(new HashSet<Map<String, Object>>() {{
        add(service);
      }});

    ingressRoute_spec.setRoutes(new HashSet<SpecRoute>() {{
        add(spec_route);
      }});

    ingressRoute.setSpec(ingressRoute_spec);

    return ingressRoute;
  }

}
