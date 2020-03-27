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

package org.apache.submarine.server.submitter.mock;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.JobSubmitter;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.exception.UnsupportedJobTypeException;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.spec.JobSpec;

public class MockSubmitter implements JobSubmitter {
  @Override
  public void initialize(SubmarineConfiguration conf) {
    System.out.println("mock");
  }

  @Override
  public String getSubmitterType() {
    return "mock";
  }

  @Override
  public Job submitJob(JobSpec jobSpec) throws UnsupportedJobTypeException, InvalidSpecException {
    return null;
  }
}
