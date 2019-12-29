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

package org.apache.submarine.server.submitter.k8s.model.tfjob;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.junit.Test;

import java.io.File;
import java.io.FileReader;
import java.net.URL;

public class TFJobTest {
  @Test
  public void testFromJson() throws Exception {
    URL fileUrl = this.getClass().getResource("/tf_job_mnist.json");
    Gson gson = new Gson();
    TFJob tfJob = gson.fromJson(new FileReader(new File(fileUrl.toURI())), TFJob.class);
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(tfJob));
  }
}
