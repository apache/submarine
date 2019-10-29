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

package org.apache.submarine.jobserver.rest.dao;


/**
 * One component contains role, count and resources.
 * The role name could be tensorflow ps, Pytorch master or tensorflow worker
 * The count is the count of the role instance
 * The resource is the memory/vcore/gpu resource strings in the format:
 * "memory=2048M,vcore=4,nvidia.com/gpu=1"
 */

public class Component {

  public String getRole() {
    return role;
  }

  public void setRole(String role) {
    this.role = role;
  }

  public String getCount() {
    return count;
  }

  public void setCount(String count) {
    this.count = count;
  }

  public String getResources() {
    return resources;
  }

  public void setResources(String resources) {
    this.resources = resources;
  }

  String role;
  String count;
  String resources;

  public Component() {}

  public Component(String r, String ct, String res) {
    this.count = ct;
    this.role = r;
    this.resources = res;
  }

}
