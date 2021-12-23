package org.apache.submarine.server.k8s.agent.util;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.core.MediaType;

import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.k8s.agent.SubmarineAgent;
import org.apache.submarine.server.rest.RestConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RestClient {
  private static final Logger LOG = LoggerFactory.getLogger(RestClient.class);
  private final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
  private Client client = ClientBuilder.newClient();
  private final String API_SERVER_URL;
  public RestClient(String host, Integer port) {
    LOG.info("SERVER_HOST:" + host);
    LOG.info("SERVER_PORT:" + port);
    API_SERVER_URL = String.format("http://%s:%d/", host, port);
  }
  
    
  public void callStatusUpdate(CustomResourceType type, String resourceId, String status) {
      LOG.info("Targeting url:" + API_SERVER_URL);
      LOG.info("Targeting uri:" + API_SERVER_URL);
      
      String uri = String.format("api/%s/%s/%s/%s/%s", RestConstants.V1,
              RestConstants.INTERNAL, type.toString(), resourceId, status);
      LOG.info("Targeting uri:" + uri);
            
      client.target(API_SERVER_URL)
      .path(uri)
      .request(MediaType.APPLICATION_JSON).post(null, String.class);        
  }
  
}
