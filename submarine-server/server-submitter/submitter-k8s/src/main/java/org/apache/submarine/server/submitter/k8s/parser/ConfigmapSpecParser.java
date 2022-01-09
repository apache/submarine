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


import io.kubernetes.client.openapi.models.V1ConfigMap;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;

public class ConfigmapSpecParser {

  private static final Logger LOG = LoggerFactory.getLogger(ConfigmapSpecParser.class);

  public static V1ConfigMap parseConfigMap(String name, String... values) {
    Map<String, String> datas = new LinkedHashMap<>();
    for (int i = 0, size = values.length; i < size; i += 2) {
      try {
        datas.put(values[i], values[i + 1]);
      } catch (ArrayIndexOutOfBoundsException e) {// Avoid values by odd numbers
        LOG.warn("Can not find ConfigMap value in index[{}], skip this value", i + 1);
      }
    }
    return parseConfigMap(name, datas);
  }

  public static V1ConfigMap parseConfigMap(String name, Map<String, String> datas) {
    V1ConfigMap configMap = new V1ConfigMap();
    /*
      Required value
      1. metadata.name
      2. spec.data
      3. spec.resources
      Others are not necessary
     */

    V1ObjectMeta metadata = new V1ObjectMeta();
    metadata.setName(name);
    configMap.setMetadata(metadata);
    configMap.data(datas);

    return configMap;
  }

}
