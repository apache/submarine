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
package org.apache.submarine.serve.istio;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.gson.annotations.SerializedName;

public class IstioHTTPMatchRequest {

  @SerializedName("uri")
  @JsonProperty("uri")
  private IstioPrefix prefix;

  public IstioHTTPMatchRequest() {
  }

  public IstioHTTPMatchRequest(String prefix) {
    this.prefix = new IstioPrefix(prefix);
  }

  public static class IstioPrefix {

    @SerializedName("prefix")
    @JsonProperty("prefix")
    private String path;

    public IstioPrefix() {
    }

    public IstioPrefix(String path){
      this.path = path;
    }

    public String getPath() {
      return path;
    }

    public void setPath(String path) {
      this.path = path;
    }
  }

  public IstioPrefix getPrefix() {
    return prefix;
  }

  public void setPrefix(IstioPrefix prefix) {
    this.prefix = prefix;
  }
}
