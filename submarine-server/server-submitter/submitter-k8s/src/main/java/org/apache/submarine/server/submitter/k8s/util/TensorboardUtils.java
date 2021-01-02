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

package org.apache.submarine.server.submitter.k8s.util;

public class TensorboardUtils {
  /*
  Prefix constants
   */
  public static final String PV_PREFIX = "tfboard-pv-";
  public static final String HOST_PREFIX = "/tmp/tfboard-logs/";
  public static final String STORAGE = "1Gi";
  public static final String PVC_PREFIX = "tfboard-pvc-";
  public static final String DEPLOY_PREFIX = "tfboard-";
  public static final String POD_PREFIX = "tfboard-";
  public static final String IMAGE_NAME = "tensorflow/tensorflow:1.11.0";
  public static final String SVC_PREFIX = "tfboard-svc-";
  public static final String INGRESS_PREFIX = "tfboard-ingressroute";
  public static final String PATH_PREFIX = "/tfboard-";
  public static final Integer DEFAULT_TENSORBOARD_PORT = 6006;
  public static final Integer SERVICE_PORT = 8080;
}
