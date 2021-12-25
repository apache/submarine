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

package org.apache.submarine.server.s3;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import javax.ws.rs.core.Response;
import io.minio.GetObjectArgs;
import io.minio.ListObjectsArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.RemoveObjectsArgs;
import io.minio.Result;
import io.minio.messages.DeleteError;
import io.minio.messages.DeleteObject;
import io.minio.messages.Item;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;


public class Client {
  public MinioClient minioClient;

  public Client() {
    minioClient = MinioClient.builder()
        .endpoint(S3Constants.ENDPOINT)
        .credentials(S3Constants.ACCESSKEY, S3Constants.SECRETKEY)
        .build();
  }

  public Client(String endpoint) {
    minioClient = MinioClient.builder()
        .endpoint(endpoint)
        .credentials(S3Constants.ACCESSKEY, S3Constants.SECRETKEY)
        .build();
  }

  /**
   * Get a list of artifact path under the experiment
   *
   * @param experimentId experiment id
   * @return a list of artifact path
   */
  public List<String> listArtifactByExperimentId(String experimentId) throws SubmarineRuntimeException {
    Iterable<Result<Item>> artifactNames = minioClient.listObjects(ListObjectsArgs.builder()
        .bucket(S3Constants.BUCKET).prefix(experimentId + "/").delimiter("/").build());
    List<String> response = new ArrayList<>();
    Iterable<Result<Item>> artifacts;
    for (Result<Item> artifactName : artifactNames) {
      try {
        artifacts = minioClient.listObjects(ListObjectsArgs.builder().bucket(S3Constants.BUCKET)
            .prefix(artifactName.get().objectName()).delimiter("/").build());
        for (Result<Item> artifact: artifacts) {
          response.add("s3://" + S3Constants.BUCKET + "/" + artifact.get().objectName());
        }
      } catch (Exception e) {
        throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          e.getMessage());
      }
    }
    return response;
  }

  /**
   * Delete all the artifacts under given experiment name
   *
   * @param experimentId experiment id
   */
  public void deleteArtifactsByExperiment(String experimentId) {
    deleteAllArtifactsByFolder(experimentId);
  }

  /**
   * Delete all the artifacts under s3://submarine
   */
  public void deleteAllArtifacts() {
    deleteAllArtifactsByFolder("");
  }

  /**
   * Download an artifact
   *
   * @param path artifact path
   * @return an array of byte
   */
  public byte[] downloadArtifact(String path) {
    byte[] buffer;
    int b;
    try (InputStream is = minioClient.getObject(
      GetObjectArgs.builder()
      .bucket(S3Constants.BUCKET)
      .object(path)
      .build())) {
      b = is.read();
      if (b == -1) {
        return new byte[0];
      }
      buffer = new byte[1 + is.available()];
      buffer[0] = (byte) b;
      int i = 1;
      while ((b = is.read()) != -1){
        buffer[i] = (byte) b;
        i += 1;
      }
    } catch (Exception e) {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          e.getMessage());
    }
    return buffer;
  }

  /**
   * Upload an artifact
   *
   * @param path path of the file
   * @param content content of the given file
   */
  public void logArtifact(String path, byte[] content) throws SubmarineRuntimeException {
    InputStream targetStream = new ByteArrayInputStream(content);
    try {
      minioClient.putObject(PutObjectArgs.builder().bucket(S3Constants.BUCKET).
          object(path).stream(targetStream,
          content.length, -1).build());
    } catch (Exception e) {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          e.getMessage());
    }
  }

  /**
   * Delete all elements under the given folder path
   */
  private void deleteAllArtifactsByFolder(String folder) {
    Iterable<Result<Item>> artifactNames = minioClient.listObjects(ListObjectsArgs.builder()
        .bucket(S3Constants.BUCKET).prefix(folder + "/").recursive(true).build());
    List<DeleteObject> objects = new LinkedList<>();
    for (Result<Item> artifactName: artifactNames){
      try {
        objects.add(new DeleteObject(artifactName.get().objectName()));
        Iterable<Result<DeleteError>> results = minioClient.removeObjects(
            RemoveObjectsArgs.builder().bucket(S3Constants.BUCKET).objects(objects).build());
        for (Result<DeleteError> result : results) {
          DeleteError error = result.get();
          throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
              "Error in deleting object " + error.objectName() + "; " + error.message());
        }
      } catch (Exception e) {
        throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
            e.getMessage());
      }
    }
  }
}
