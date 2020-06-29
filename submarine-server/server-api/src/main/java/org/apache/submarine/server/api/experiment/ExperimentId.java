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

package org.apache.submarine.server.api.experiment;

/**
 * The unique id for experiment. Formatter: experiment_${server_timestamp}_${counter}
 * Such as: experiment_1577627710_0001
 */
public class ExperimentId implements Comparable<ExperimentId> {
  private static final String EXPERIMENT_ID_PREFIX = "experiment_";
  private static final int EXPERIMENT_ID_MIN_DIGITS = 4;

  private int id;

  private long serverTimestamp;

  /**
   * Get the object of JobId.
   * @param jobId job id string
   * @return object
   */
  public static ExperimentId fromString(String jobId) {
    if (jobId == null) {
      return null;
    }
    String[] components = jobId.split("\\_");
    if (components.length != 3) {
      return null;
    }
    return ExperimentId.newInstance(Long.parseLong(components[1]), Integer.parseInt(components[2]));
  }

  /**
   * Ge the object of JobId.
   * @param serverTimestamp the timestamp when the server start
   * @param id count
   * @return object
   */
  public static ExperimentId newInstance(long serverTimestamp, int id) {
    ExperimentId experimentId = new ExperimentId();
    experimentId.setServerTimestamp(serverTimestamp);
    experimentId.setId(id);
    return experimentId;
  }

  /**
   * Get the count of job since the server started
   * @return number
   */
  public int getId() {
    return id;
  }

  /**
   * Set the count of job
   * @param id number
   */
  public void setId(int id) {
    this.id = id;
  }

  /**
   * Get the timestamp(s) when the server started
   * @return timestamp(s)
   */
  public long getServerTimestamp() {
    return serverTimestamp;
  }

  /**
   * Set the server started timestamp(s)
   * @param timestamp seconds
   */
  public void setServerTimestamp(long timestamp) {
    this.serverTimestamp = timestamp;
  }

  @Override
  public int compareTo(ExperimentId o) {
    return this.getId() > o.getId() ? 1 : 0;
  }

  @Override
  public int hashCode() {
    final int prime = 371237;
    int result = 6521;
    result = prime * result + (int) (serverTimestamp ^ (serverTimestamp >>> 32));
    result = prime * result + getId();
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    if (this == obj) {
      return true;
    }
    ExperimentId other = (ExperimentId) obj;
    if (this.getServerTimestamp() != other.getServerTimestamp()) {
      return false;
    }
    return this.getId() == other.getId();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(64);
    sb.append(EXPERIMENT_ID_PREFIX).append(serverTimestamp).append("_");
    format(sb, getId());
    return sb.toString();
  }

  private void format(StringBuilder sb, long value) {
    int minimumDigits = EXPERIMENT_ID_MIN_DIGITS;
    if (value < 0) {
      sb.append('-');
      value = -value;
    }

    long tmp = value;
    do {
      tmp /= 10;
    } while (--minimumDigits > 0 && tmp > 0);

    for (int i = minimumDigits; i > 0; --i) {
      sb.append('0');
    }
    sb.append(value);
  }
}
