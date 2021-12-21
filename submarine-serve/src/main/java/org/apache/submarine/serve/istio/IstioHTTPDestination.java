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

public class IstioHTTPDestination {
  @SerializedName("destination")
  private IstioDestination destination;

  public IstioHTTPDestination(String host){
    this.destination = new IstioDestination(host);
  }


  public static class IstioDestination{
    @SerializedName("host")
    private String host;

    @SerializedName("port")
    private IstioPort port;

    public IstioDestination(String host) {
      this.host = host;
      this.port = new IstioPort(IstioConstants.DEFAULT_SERVE_POD_PORT);
    }
  }

  public static class IstioPort {
    @SerializedName("number")
    private Integer number;

    public IstioPort(Integer port){
      this.number = port;
    }
  }
}
