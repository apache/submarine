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

package org.apache.submarine.server.api;

import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.exception.UnsupportedJobTypeException;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.spec.JobSpec;

/**
 * Handle the job's operate (CRUD)
 */
public interface JobHandler {
  /**
   * Submit job
   * @param jobSpec job spec
   * @return job object
   * @throws UnsupportedJobTypeException caused by the unsupported job type
   */
  Job submitJob(JobSpec jobSpec) throws UnsupportedJobTypeException, InvalidSpecException;

  /**
   * Get job info
   * @param jobSpec job spec
   * @return job object
   * @throws UnsupportedJobTypeException caused by the unsupported job type
   */
  default Job getJob(JobSpec jobSpec) throws UnsupportedJobTypeException {
    // TODO(submarine) should implementing later
    return null;
  }

  /**
   * Update job info
   * @param jobSpec job spec
   * @return job object
   * @throws UnsupportedJobTypeException caused by the unsupported job type
   */
  default Job updateJob(JobSpec jobSpec) throws UnsupportedJobTypeException {
    // TODO(submarine) should implementing later
    return null;
  }

  /**
   * Delete the specified job
   * @param jobSpec job spec
   * @return job object
   * @throws UnsupportedJobTypeException caused by the unsupported job type
   */
  default Job deleteJob(JobSpec jobSpec) throws UnsupportedJobTypeException {
    // TODO(submarine) should implementing later
    return null;
  }
}
