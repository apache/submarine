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

package org.apache.submarine.server.api.spec.code;

/**
 * Describes some of the main variable properties of git code
 */
public class GitCodeSpec {

  /**
   * Default git branch.
   * The new git branch has changed from master to main
   */
  public static final String DEFAULT_BRANCH = "main";

  private final String url;

  private String branch = DEFAULT_BRANCH;

  private String username;

  private String password;

  /**
   * Whether the git-sync should trust a self-signed certificate
   */
  private Boolean trustCerts;

  public GitCodeSpec(String url) {
    this.url = url;
  }

  public String getUrl() {
    return url;
  }

  public String getBranch() {
    return branch == null || branch.isBlank() ? DEFAULT_BRANCH : branch;
  }

  public void setBranch(String branch) {
    this.branch = branch;
  }

  public String getUsername() {
    return username;
  }

  public void setUsername(String username) {
    this.username = username;
  }

  public String getPassword() {
    return password;
  }

  public void setPassword(String password) {
    this.password = password;
  }

  public Boolean getTrustCerts() {
    return trustCerts;
  }

  public void setTrustCerts(Boolean trustCerts) {
    this.trustCerts = trustCerts;
  }
}
