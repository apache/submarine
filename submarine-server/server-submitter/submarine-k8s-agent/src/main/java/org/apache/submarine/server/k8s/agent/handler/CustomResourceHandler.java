package org.apache.submarine.server.k8s.agent.handler;

import java.io.IOException;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.util.Config;

public abstract class CustomResourceHandler {
    private CoreV1Api coreApi;
    private ApiClient client = null;  
    private String namespace;
    private String crType;
    private String crName;
    
    public CustomResourceHandler() throws IOException {
        this.client = Config.defaultClient();
        Configuration.setDefaultApiClient(client);
        this.coreApi = new CoreV1Api(this.client);
    }
    
    public abstract void init(String namespace, String crType, String crName);
    public abstract void run();
    public abstract void onAddEvent();
    public abstract void onModifyEvent();
    public abstract void onDeleteEvent();

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
    
    
    
}
