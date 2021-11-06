package org.apache.submarine.server.s3;

import org.junit.Test;
import java.util.List;


public class ClientTest {
  private Client client = new Client();
  private final String testExperimentId = "experiment-1636102552127-0001";
  @Test
  public void testListArtifactByExperimentId() {
    List<String> results = client.listArtifactByExperimentId(testExperimentId);
    System.out.println(results);
  }
}
