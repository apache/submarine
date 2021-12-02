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

package org.apache.submarine.server.submitter.k8s;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

import io.kubernetes.client.openapi.models.V1DeleteOptions;
import io.kubernetes.client.openapi.models.V1JobCondition;
import io.kubernetes.client.openapi.models.V1JobConditionBuilder;
import io.kubernetes.client.openapi.models.V1JobStatus;
import io.kubernetes.client.openapi.models.V1JobStatusBuilder;
import io.kubernetes.client.openapi.models.V1Status;
import io.kubernetes.client.openapi.models.V1StatusBuilder;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.apache.submarine.server.submitter.k8s.parser.ExperimentSpecParser;
import org.joda.time.DateTime;
import org.junit.Assert;
import org.junit.Test;

public class MLJobConverterTest extends SpecBuilder {
  @Test
  public void testMLJob2Job() throws IOException, URISyntaxException, InvalidSpecException {
    // Accepted Status
    ExperimentSpec spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    MLJob mlJob = ExperimentSpecParser.parseJob(spec);
    V1JobStatus status = new V1JobStatusBuilder().build();
    mlJob.setStatus(status);
    Experiment experiment = MLJobConverter.toJobFromMLJob(mlJob);
    Assert.assertNull(experiment.getStatus());
    Assert.assertNull(experiment.getStatus());

    // Created Status
    DateTime startTime = new DateTime();
    mlJob.getStatus().setStartTime(startTime);

    List<V1JobCondition> conditions = new ArrayList<>();
    DateTime createdTime = new DateTime();
    V1JobCondition condition = new V1JobConditionBuilder().withStatus("True")
        .withType("Created").withLastTransitionTime(createdTime).build();
    conditions.add(condition);
    mlJob.getStatus().setConditions(conditions);

    experiment = MLJobConverter.toJobFromMLJob(mlJob);
    Assert.assertEquals(Experiment.Status.STATUS_CREATED.getValue(), experiment.getStatus());
    Assert.assertEquals(startTime.toString(), experiment.getCreatedTime());

    // Running Status
    DateTime runningTime = new DateTime();
    condition = new V1JobConditionBuilder().withStatus("True")
        .withType("Running").withLastTransitionTime(runningTime).build();
    conditions.add(condition);

    mlJob.getStatus().setConditions(conditions);
    experiment = MLJobConverter.toJobFromMLJob(mlJob);
    Assert.assertEquals(Experiment.Status.STATUS_RUNNING.toString(), experiment.getStatus());
    Assert.assertEquals(runningTime.toString(), experiment.getRunningTime());

    // Succeeded Status
    DateTime finishedTime = new DateTime();
    mlJob.getStatus().setCompletionTime(finishedTime);
    experiment = MLJobConverter.toJobFromMLJob(mlJob);
    Assert.assertEquals(Experiment.Status.STATUS_SUCCEEDED.toString(), experiment.getStatus());
    Assert.assertEquals(finishedTime.toString(), experiment.getFinishedTime());
  }

  @Test
  public void testStatus2Job() {
    V1Status status = new V1StatusBuilder().withStatus("Success").build();
    Experiment experiment = MLJobConverter.toJobFromStatus(status);
    Assert.assertNotNull(experiment);
    Assert.assertEquals(Experiment.Status.STATUS_DELETED.getValue(), experiment.getStatus());
  }

  @Test
  public void testMLJob2DeleteOptions() throws IOException, URISyntaxException,
      InvalidSpecException {
    ExperimentSpec spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, tfJobReqFile);
    MLJob mlJob = ExperimentSpecParser.parseJob(spec);
    V1DeleteOptions options = MLJobConverter.toDeleteOptionsFromMLJob(mlJob);
    Assert.assertNotNull(options);
    Assert.assertEquals(mlJob.getApiVersion(), options.getApiVersion());
  }
}
