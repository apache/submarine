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

package org.apache.submarine.server.submitter.k8s.model.middlewares;

import java.util.List;

import com.google.gson.annotations.SerializedName;


public class StripPrefix {

  @SerializedName("prefixes")
  private List<String> prefixes;

  @SerializedName("forceSlash")
  private Boolean forceSlash;
  

  public StripPrefix() {
    forceSlash = true; // default to true
  }

  public StripPrefix(List<String> prefixes, Boolean forceSlash) {
    this.prefixes = prefixes;
    this.forceSlash = forceSlash;
  }

  public List<String> getPrefixes() {
    return this.prefixes;
  }

  public void setPrefixes(List<String> prefixes) {
    this.prefixes = prefixes;
  }

  public Boolean isForceSlash() {
    return this.forceSlash;
  }

  public Boolean getForceSlash() {
    return this.forceSlash;
  }

  public void setForceSlash(Boolean forceSlash) {
    this.forceSlash = forceSlash;
  }

  public StripPrefix prefixes(List<String> prefixes) {
    setPrefixes(prefixes);
    return this;
  }

  public StripPrefix forceSlash(Boolean forceSlash) {
    setForceSlash(forceSlash);
    return this;
  }

  @Override
  public String toString() {
    return "{" +
      " prefixes='" + getPrefixes() + "'" +
      ", forceSlash='" + isForceSlash() + "'" +
      "}";
  }
  
}
