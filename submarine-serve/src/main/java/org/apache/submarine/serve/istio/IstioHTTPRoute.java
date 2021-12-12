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

import com.google.gson.annotations.SerializedName;
import org.apache.submarine.serve.utils.IstioConstants;

import java.util.ArrayList;
import java.util.List;

public class IstioHTTPRoute {
  @SerializedName("match")
  private List<IstioHTTPMatchRequest> match = new ArrayList<>();

  @SerializedName("route")
  private List<IstioHTTPDestination> route = new ArrayList<>();

  @SerializedName("rewrite")
  private IstioRewrite rewrite;

  public IstioHTTPRoute() {
    this.rewrite = new IstioRewrite(IstioConstants.REWRITE_URL);
  }

  @Override
  public String toString() {
    return "'rewrite': {'uri': " + rewrite.getRewrite() + "}";

  }

  public void addHTTPMatchRequest(IstioHTTPMatchRequest match){
    this.match.add(match);
  }

  public void addHTTPDestination(IstioHTTPDestination destination){
    this.route.add(destination);
  }
  public static class IstioRewrite{
    @SerializedName("uri")
    private String uri;

    public IstioRewrite(String rewrite){
      this.uri = rewrite;
    }

    public String getRewrite() {
      return uri;
    }
  }

}
