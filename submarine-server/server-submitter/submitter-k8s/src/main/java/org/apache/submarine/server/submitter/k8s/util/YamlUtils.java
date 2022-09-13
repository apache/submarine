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

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.dataformat.yaml.YAMLGenerator;
import com.fasterxml.jackson.dataformat.yaml.YAMLMapper;
import com.fasterxml.jackson.datatype.joda.JodaModule;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

/**
 * Support the conversion of objects to yaml format,
 * so that the resource information can be displayed on the log in a more k8s declarative and readable manner
 */
public class YamlUtils {

  private static final YAMLMapper YAML_MAPPER = YAMLMapper.builder(new YAMLFactory()
                  .disable(YAMLGenerator.Feature.USE_NATIVE_TYPE_ID))
          .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
          .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)
          .disable(SerializationFeature.FAIL_ON_EMPTY_BEANS)
          .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
          .disable(YAMLGenerator.Feature.WRITE_DOC_START_MARKER)
          .build();

  static {
    YAML_MAPPER.registerModule(new JavaTimeModule())
            .registerModule(new JodaModule())
            .setSerializationInclusion(JsonInclude.Include.NON_NULL);
  }

  /**
   * Pretty yaml
   */
  public static String toPrettyYaml(Object pojoObject) {
    try {
      return YAML_MAPPER.writeValueAsString(pojoObject);
    } catch (JsonProcessingException ex) {
      throw new RuntimeException("Parse yaml failed! " + pojoObject.getClass().getName(), ex);
    }
  }

  /**
   * Read yaml to class
   */
  public static <T> T readValue(String content, Class<T> tClass) {
    try {
      return YAML_MAPPER.readValue(content, tClass);
    } catch (JsonProcessingException ex) {
      throw new RuntimeException("Read yaml failed! " + tClass.getName(), ex);
    }
  }
}
