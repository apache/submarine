package org.apache.submarine.server.k8s.agent;

import java.io.IOException;

import org.apache.submarine.server.k8s.agent.bean.CustomResourceType;
import org.apache.submarine.server.k8s.agent.handler.CustomResourceHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.util.Config;
import io.kubernetes.client.util.Watch;
import io.kubernetes.client.util.Watchable;
import io.kubernetes.client.util.generic.GenericKubernetesApi;


public class SubmarineAgent {
    private static final Logger LOG = LoggerFactory.getLogger(SubmarineAgent.class);
    private String namespace;
    private String customResourceType;
    private String customResourceName;
    private CustomResourceType type;
    private CustomResourceHandler handler;
    
    
    public SubmarineAgent(String namespace, String customResourceType, String customResourceName) throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException {
        this.namespace = namespace;
        this.customResourceType = customResourceType;
        this.customResourceName = customResourceName;
        this.type = CustomResourceType.valueOf(customResourceType);
        this.handler = HandlerFactory.getHandler(this.type);

    }
    
    public void start() {
        
    }
    
    
    public static void main(String[] args) throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException {
        String namespace = System.getenv("NAMESPACE");
        String customResourceType = System.getenv("CUSTOM_RESOURCE_TYPE");
        String customResourceName = System.getenv("CUSTOM_RESOURCE_NAME");
        LOG.info(String.format("NAMESPACE:%s", customResourceType));
        LOG.info(String.format("CUSTOM_RESOURCE_TYPE:%s", customResourceType));
        LOG.info(String.format("CUSTOM_RESOURCE_NAME:%s", customResourceName));
        
        SubmarineAgent agent = new SubmarineAgent(customResourceType, customResourceType, customResourceName);
        agent.start();
        
    }
    

}
