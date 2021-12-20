package org.apache.submarine.server.s3;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import java.util.List;


public class ClientTest {
  private Client client = new Client();
  private final String testExperimentId = "experiment-sample";
  private final String bucket = "s3://submarine";

  @After
  public void cleanAll() {
    client.deleteAllArtifacts();
  }

  @Test
  public void testLogArtifactAndDownloadArtifact() {
    String path = "sample_folder/sample_file";
    byte[] content = "0123456789".getBytes();;
    client.logArtifact(path, content);
    byte[] response = client.downloadArtifact(path);
    Assert.assertArrayEquals(content, response);
  }

  @Test
  public void testListArtifactByExperimentIdAndDeleteArtifactByExperiment() {
    String testModelName  = "sample";
    byte[] content = "0123456789".getBytes();

    String[] artifactPaths = {
        testExperimentId + "/" + testModelName + "/1",
        testExperimentId + "/" + testModelName + "/2"};
    String[] actualResults = {
        bucket + "/" + testExperimentId + "/" + testModelName + "/1",
        bucket + "/" + testExperimentId + "/" + testModelName + "/2"};
    client.logArtifact(artifactPaths[0], content);
    client.logArtifact(artifactPaths[1], content);
    List<String> results = client.listArtifactByExperimentId(testExperimentId);
    Assert.assertArrayEquals(actualResults, results.toArray());

    client.deleteArtifactsByExperiment(testExperimentId);
    results = client.listArtifactByExperimentId(testExperimentId);
    Assert.assertArrayEquals(new String[0], results.toArray());
  }
}
