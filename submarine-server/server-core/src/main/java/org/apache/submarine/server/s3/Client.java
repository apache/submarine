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
import java.util.Stack;
import javax.ws.rs.core.Response;
import java.util.Map;
import java.util.HashMap;

import io.minio.CopyObjectArgs;
import io.minio.CopySource;
import io.minio.GetObjectArgs;
import io.minio.ListObjectsArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.RemoveObjectsArgs;
import io.minio.Result;
import io.minio.messages.DeleteError;
import io.minio.messages.DeleteObject;
import io.minio.messages.Item;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;

/**
 * S3(Minio) default client
 */
public enum Client {
  DEFAULT(SubmarineConfiguration.getInstance().getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_ENDPOINT)),
  CUSTOMER("http://localhost:9000");

  /* submarine config */
  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  /* minio client */
  private final MinioClient minioClient;

  public static Map<String, Client> clientFactory = new HashMap<String, Client>();
  private final String endpoint;

  static {
    for (Client clientSingleton : Client.values()) {
      clientFactory.put(clientSingleton.endpoint, clientSingleton);
    }
  }

  private Client(String endpoint) {
    this.endpoint = endpoint;
    this.minioClient =  MinioClient.builder()
                        .endpoint(endpoint)
                        .credentials(
                          conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_ACCESS_KEY_ID),
                          conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_SECRET_ACCESS_KEY)
                        ).build();
  }

  private Client() {
    this.endpoint = conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_ENDPOINT);
    this.minioClient =  MinioClient.builder()
                        .endpoint(conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_ENDPOINT))
                        .credentials(
                            conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_ACCESS_KEY_ID),
                            conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_S3_SECRET_ACCESS_KEY)
                        ).build();
  }

  public static Client getInstance() {
    return clientFactory.get(SubmarineConfVars.ConfVars.SUBMARINE_S3_ENDPOINT.varValue);
  }

  public static Client getInstance(String endpoint) {
    try {
      return clientFactory.get(endpoint);
    } catch (Exception e) {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          e.getMessage());
    }
  }

  /**
   * Get a list of artifact path under the experiment.
   *
   * @param path path of the artifact directory
   * @return a list of artifact path
   */
  public List<String> listArtifact(String path) throws SubmarineRuntimeException {
    try {
      Iterable<Result<Item>> artifacts = minioClient.listObjects(ListObjectsArgs.builder()
          .bucket(S3Constants.BUCKET).prefix(path + "/").delimiter("/").build());
      List<String> response = new ArrayList<>();
      for (Result<Item> artifact: artifacts) {
        response.add("s3://" + S3Constants.BUCKET + "/" + artifact.get().objectName());
      }
      return response;
    } catch (Exception e) {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          e.getMessage());
    }
  }

  /**
   * Delete all the artifacts under given experiment name.
   *
   * @param experimentId experiment id
   */
  public void deleteArtifactsByExperiment(String experimentId) {
    deleteAllArtifactsByFolder(String.format("experiment/%s", experimentId));
  }

  /**
   * Delete all the artifacts under s3://submarine.
   */
  public void deleteAllArtifacts() {
    deleteAllArtifactsByFolder("");
  }

  /**
   * Delete all the artifacts under given experiment name.
   */
  public void deleteArtifactsByModelVersion(String modelName, Integer version, String modelId) {
    // the directory of storing a single model must be unique for serving
    String uniqueModelPath = String.format("%s-%d-%s", modelName, version, modelId);
    deleteAllArtifactsByFolder(String.format("registry/%s", uniqueModelPath));
  }


  /**
   * Download an artifact.
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
   * Copy an artifact.
   *
   * @param targetPath path of the target file
   * @param sourcePath path of the source file
   */
  public void copyArtifact(String targetPath, String sourcePath) {
    try {
      minioClient.copyObject(CopyObjectArgs.builder()
          .bucket(S3Constants.BUCKET)
          .object(targetPath)
          .source(CopySource.builder()
              .bucket(S3Constants.BUCKET)
              .object(sourcePath)
              .build())
          .build());
    } catch (Exception e) {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          e.getMessage());
    }
  }

  /**
   * Upload an artifact.
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

  public List<String> listAllObjects(String path) throws SubmarineRuntimeException {
    List<String> result = new ArrayList<>();
    Stack<String> dirs = new Stack<>();
    dirs.add(path);
    while (!dirs.empty()) {
      String dir = dirs.pop();
      try {
        Iterable<Result<Item>> artifacts = minioClient.listObjects(ListObjectsArgs.builder()
            .bucket(S3Constants.BUCKET).prefix(dir).delimiter("/").build());
        for (Result<Item> artifact: artifacts) {
          String objectName = artifact.get().objectName();
          if (objectName.endsWith("/")) {
            dirs.add(objectName);
          } else {
            result.add(objectName);
          }
        }
      } catch (Exception e) {
        throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
            e.getMessage());
      }
    }

    return result;
  }

  /**
   * Delete all elements under the given folder path.
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
