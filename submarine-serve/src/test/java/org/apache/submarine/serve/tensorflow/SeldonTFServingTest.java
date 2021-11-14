// /*
//  * Licensed to the Apache Software Foundation (ASF) under one
//  * or more contributor license agreements.  See the NOTICE file
//  * distributed with this work for additional information
//  * regarding copyright ownership.  The ASF licenses this file
//  * to you under the Apache License, Version 2.0 (the
//  * "License"); you may not use this file except in compliance
//  * with the License.  You may obtain a copy of the License at
//  *
//  *   http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing,
//  * software distributed under the License is distributed on an
//  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//  * KIND, either express or implied.  See the License for the
//  * specific language governing permissions and limitations
//  * under the License.
//  */
// package org.apache.submarine.serve.tensorflow;

// import io.kubernetes.client.ApiClient;
// import io.kubernetes.client.Configuration;
// import io.kubernetes.client.apis.AppsV1Api;
// import io.kubernetes.client.apis.CustomObjectsApi;
// import io.kubernetes.client.util.ClientBuilder;
// import io.kubernetes.client.util.KubeConfig;
// import org.junit.Before;
// import org.junit.Test;
// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;

// import java.io.FileReader;
// import java.io.IOException;

// public class SeldonTFServingTest {
//   private static final Logger LOG = LoggerFactory.getLogger(SeldonTFServingTest.class);

//   private static CustomObjectsApi k8sApi;

//   private AppsV1Api appsV1Api;
//   @Before
//   public void startUp() throws IOException {
//     String confPath = System.getProperty("user.home") + "/.kube/config";
//     KubeConfig config = KubeConfig.loadKubeConfig(new FileReader(confPath));
//     ApiClient client = ClientBuilder.kubeconfig(config).build();
//     Configuration.setDefaultApiClient(client);
//     k8sApi = new CustomObjectsApi();
//     if (appsV1Api == null) {
//       appsV1Api = new AppsV1Api();
//     }
//   }
//   @Test
//   public void test() {
//     this.create();
//   }
//   private void create() {
// //    SeldonTFServing seldonTFServing = new SeldonTFServing("simple", 1);
// //    try {
// //      Object create = k8sApi.createNamespacedCustomObject(seldonTFServing.getGroup(),
// //              seldonTFServing.getVersion(),
// //              "default",
// //              seldonTFServing.getPlural(),
// //              seldonTFServing,
// //              "true");
// //    } catch (ApiException e){
// //      LOG.error(e.getMessage(), e);
// //      throw new SubmarineRuntimeException("error");
// //    }
//   }
// }
