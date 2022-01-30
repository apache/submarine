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

package org.apache.submarine.server.k8s.agent.handler;

import java.io.FileReader;
import java.io.IOException;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.k8s.agent.util.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.apis.CustomObjectsApi;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import okhttp3.OkHttpClient;

public abstract class CustomResourceHandler {
    private static final Logger LOG = LoggerFactory.getLogger(CustomResourceHandler.class);
    private static final String KUBECONFIG_ENV = "KUBECONFIG";
    
    protected ApiClient client = null;  
    protected CustomObjectsApi customObjectsApi = null;
    protected CoreV1Api coreV1Api = null;
    protected String namespace;
    protected String crType;
    protected String crName;
    protected String serverHost;
    protected Integer serverPort;
    protected String resourceId;
    protected RestClient restClient;
    
    public CustomResourceHandler() throws IOException {
      try {
        String path = System.getenv(KUBECONFIG_ENV);
        LOG.info("PATH:" + path);
        KubeConfig config = KubeConfig.loadKubeConfig(new FileReader(path));
        client = ClientBuilder.kubeconfig(config).build();
      } catch (Exception e) {
        LOG.info("Maybe in cluster mode, try to initialize the client again.");
        try {
          client = ClientBuilder.cluster().build();
        } catch (IOException e1) {
           LOG.error("Initialize K8s submitter failed. " + e.getMessage(), e1);
           throw new SubmarineRuntimeException(500, "Initialize K8s submitter failed.");
        }
      } finally {
        // let watcher can wait until the next change
        client.setReadTimeout(0);
        OkHttpClient httpClient = client.getHttpClient();
        this.client.setHttpClient(httpClient);
        Configuration.setDefaultApiClient(client);
      }
      
      customObjectsApi = new CustomObjectsApi(client);
      coreV1Api = new CoreV1Api(client);
    }
    
    public abstract void init(String serverHost, Integer serverPort,
            String namespace, String crName, String resourceId);
    public abstract void run();

    public String getNamespace() {
        return namespace;
    }

    public void setNamespace(String namespace) {
        this.namespace = namespace;
    }

    public String getCrType() {
        return crType;
    }

    public void setCrType(String crType) {
        this.crType = crType;
    }

    public String getCrName() {
        return crName;
    }

    public void setCrName(String crName) {
        this.crName = crName;
    }

    public String getServerHost() {
        return serverHost;
    }

    public void setServerHost(String serverHost) {
        this.serverHost = serverHost;
    }

    public Integer getServerPort() {
        return serverPort;
    }

    public void setServerPort(Integer serverPort) {
        this.serverPort = serverPort;
    }

    public RestClient getRestClient() {
        return restClient;
    }

    public void setRestClient(RestClient restClient) {
        this.restClient = restClient;
    }
    
}
