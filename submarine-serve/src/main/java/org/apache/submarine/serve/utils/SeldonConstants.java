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

public class SeldonConstants {
  public static final String API_VERSION = "machinelearning.seldon.io/v1";

  public static final String KIND = "SeldonDeployment";

  public static final String GROUP = "machinelearning.seldon.io";

  public static final String VERSION = "v1";

  public static final String PLURAL = "seldondeployments";

  public static final String ENV_SECRET_REF_NAME = "submarine-serve-secret";

  public static final String SELDON_PROTOCOL = "seldon";

  public static final String KFSERVING_PROTOCOL = "kfserving";

  // TensorFlow
  public static final String TFSERVING_IMPLEMENTATION = "TENSORFLOW_SERVER";

  // PyTorch
  public static final String TRITON_IMPLEMENTATION = "TRITON_SERVER";
}
