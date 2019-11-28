/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.client.cli;

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;

import java.io.IOException;

/**
 * This class is responsible for parsing resource string and creating a
 * Resource.
 */
public class RoleResourceParser {
  private ClientContext clientContext;

  public RoleResourceParser(ClientContext clientContext) {
    this.clientContext = clientContext;
  }

  public Resource parseResource(String resourceKey, String resourceStr)
      throws ParseException, YarnException, IOException {
    return parseResourceInternal(1, resourceKey, resourceStr);
  }

  public Resource parseResource(int numberOfRoleInstances,
                                String resourceKey, String resourceStr)
      throws YarnException, ParseException, IOException {
    return parseResourceInternal(numberOfRoleInstances, resourceKey,
        resourceStr);
  }

  private Resource parseResourceInternal(int numberOfRoleInstances,
                                         String resourceKey, String resourceStr)
      throws ParseException, YarnException, IOException {
    if (numberOfRoleInstances > 0) {
      if (resourceStr == null) {
        throw new ParseException("--" + resourceKey + " is absent.");
      }
      return ResourceUtils.createResourceFromString(resourceStr,
          clientContext.getOrCreateYarnClient().getResourceTypeInfo());
    }
    return null;
  }
}