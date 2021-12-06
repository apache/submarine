package org.apache.submarine.server.k8s.agent.bean;

public enum CustomResourceType {
    TFJob("tfJob"), PYTORCHJob("pytorchJob"), Notebook("notebook");
    
    private String customResourceType;
    
    CustomResourceType(String customResourceType) {
        this.customResourceType = customResourceType; 
    }
    
    public String getCustomResourceType() {
        return this.customResourceType;
    }
    
}
