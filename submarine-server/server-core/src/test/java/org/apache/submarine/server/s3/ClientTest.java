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

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;


public class ClientTest {
  private final Client client = Client.getInstance();
  private final String testExperimentId = "experiment-sample";

  @After
  public void cleanAll() {
    client.deleteAllArtifacts();
  }

  @Test
  public void testLogArtifactAndDownloadArtifact() {
    String path = "sample_folder/sample_file";
    byte[] content = "0123456789".getBytes();
    client.logArtifact(path, content);
    byte[] response = client.downloadArtifact(path);
    Assert.assertArrayEquals(content, response);
  }

  @Test
  public void testListAndDeleteArtifactByExperimentId() {
    byte[] content = "0123456789".getBytes();

    String[] artifactPaths = {
        String.format("experiment/%s/1", testExperimentId),
        String.format("experiment/%s/2", testExperimentId)
    };
    String[] actualResults = {
        String.format("s3://%s/experiment/%s/1", S3Constants.BUCKET, testExperimentId),
        String.format("s3://%s/experiment/%s/2", S3Constants.BUCKET, testExperimentId)
    };
    client.logArtifact(artifactPaths[0], content);
    client.logArtifact(artifactPaths[1], content);
    List<String> results = client.listArtifact(String.format("experiment/%s", testExperimentId));
    Assert.assertArrayEquals(actualResults, results.toArray());

    client.deleteArtifactsByExperiment(testExperimentId);
    results = client.listArtifact(testExperimentId);
    Assert.assertArrayEquals(new String[0], results.toArray());
  }

  @Test
  public void testCopyObject() {
    String path = "sample_folder/sample_file";
    byte[] content = "0123456789".getBytes();
    client.logArtifact(path, content);
    byte[] response = client.downloadArtifact(path);
    Assert.assertArrayEquals(content, response);

    String copyPath = "sample_folder_copy/sample_file";
    client.copyArtifact(copyPath, path);
    response = client.downloadArtifact(copyPath);
    Assert.assertArrayEquals(content, response);
  }

  @Test
  public void testSingleton() {
    Client testClient = Client.getInstance();
    Assert.assertEquals(testClient, client);
  }
}
