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
package org.apache.submarine.serve.utils;

public class IstioConstants {
  public static final String API_VERSION = "networking.istio.io/v1beta1";

  public static final String KIND = "VirtualService";

  public static final String GROUP = "networking.istio.io";

  public static final String VERSION = "v1beta1";

  public static final String PLURAL = "virtualservices";

  public static final String REWRITE_URL = "/"; 

  public static final String DEFAULT_NAMESPACE = "default";

  public static final String DEFAULT_GATEWAY = "istio-system/seldon-gateway";

  public static final Integer DEFAULT_SERVE_POD_PORT = 8000;

  public static final String DEFAULT_INGRESS_HOST = "*"; 

}
