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

import io.kubernetes.client.models.V1Deployment;
import io.kubernetes.client.models.V1Service;
import junit.framework.TestCase;
import org.apache.submarine.server.submitter.k8s.model.ingressroute.IngressRoute;
import org.apache.submarine.server.submitter.k8s.util.TensorboardUtils;
import org.junit.Assert;
import org.junit.Test;

public class TensorboardSpecParserTest extends TestCase {

  @Test
  public void testParseDeployment() {
    final String id = "123456789";

    final String name = TensorboardUtils.DEPLOY_PREFIX + id;
    final String image = TensorboardUtils.IMAGE_NAME;
    final String routePath = TensorboardUtils.PATH_PREFIX + id;
    final String pvc = TensorboardUtils.PVC_PREFIX + id;

    V1Deployment deployment = TensorboardSpecParser.parseDeployment(name, image, routePath, pvc);

    Assert.assertEquals(name, deployment.getMetadata().getName());
    Assert.assertEquals(image,
        deployment.getSpec().getTemplate().getSpec().getContainers().get(0).getImage());

    Assert.assertEquals(pvc,
        deployment.getSpec().getTemplate().getSpec().getVolumes()
          .get(0).getPersistentVolumeClaim().getClaimName());
  }

  @Test
  public void testParseService() {
    final String id = "123456789";

    final String svcName = TensorboardUtils.SVC_PREFIX + id;
    final String podName = TensorboardUtils.DEPLOY_PREFIX + id;

    V1Service svc = TensorboardSpecParser.parseService(svcName, podName);

    Assert.assertEquals(svcName, svc.getMetadata().getName());
    Assert.assertEquals(podName, svc.getSpec().getSelector().get("app"));
  }

  @Test
  public void testParseIngressRoute() {
    final String id = "123456789";
    final String namespace = "default";

    final String ingressName = TensorboardUtils.INGRESS_PREFIX + id;
    final String routePath = TensorboardUtils.PATH_PREFIX + id;
    final String svcName = TensorboardUtils.SVC_PREFIX + id;

    IngressRoute ingressRoute = TensorboardSpecParser.parseIngressRoute(
        ingressName, namespace, routePath, svcName
      );

    Assert.assertEquals(ingressRoute.getMetadata().getName(), ingressName);
  }
}
