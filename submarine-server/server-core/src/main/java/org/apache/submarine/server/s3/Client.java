package org.apache.submarine.server.s3;


import java.util.ArrayList;
import java.util.List;
import io.minio.ListObjectsArgs;
import io.minio.MinioClient;
import io.minio.Result;
import io.minio.messages.Item;


public class Client {
  MinioClient minioClient =
      MinioClient.builder()
          .endpoint("http://submarine-minio-service:9000")
          .credentials("submarine_minio", "submarine_minio")
          .build();

  /**
   * Get a list of artifact path under the experiment
   *
   * @param id experiment id
   * @return a list of artifact path
   */
  public List<String> listArtifactByExperimentId(String id) {
    Iterable<Result<Item>> artifactNames = minioClient.listObjects(ListObjectsArgs.builder()
        .bucket("submarine").prefix(id + "/").delimiter("/").build()); //

    List<String> response = new ArrayList<>();
    Iterable<Result<Item>> artifacts;
    for (Result<Item> artifactName : artifactNames) {
      try {
        artifacts = minioClient.listObjects(ListObjectsArgs.builder().bucket("submarine")
            .prefix(artifactName.get().objectName()).delimiter("/").build());
        for (Result<Item> artifact: artifacts) {
          response.add("s3://submarine/" + artifact.get().objectName());
        }
      } catch (Exception e ) {
        System.out.println(e.getMessage());
      }
    }
    return response;
  }
}
