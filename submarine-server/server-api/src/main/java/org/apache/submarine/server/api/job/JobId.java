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

package org.apache.submarine.server.api.job;

public class JobId implements Comparable<JobId> {
  private static final String jobIdStrPrefix = "job";
  private static final String JOB_ID_PREFIX = jobIdStrPrefix + '_';
  private static final int JOB_ID_MIN_DIGITS = 4;

  private int id;
  private long serverTimestamp;

  public static JobId newInstance(long serverTimestamp, int id) {
    JobId jobId = new JobId();
    jobId.setId(id);
    return jobId;
  }

  public void setId(int id) {
    this.id = id;
  }

  public int getId() {
    return id;
  }

  public void setServerTimestamp(long timestamp) {
    this.serverTimestamp = timestamp;
  }

  public long getServerTimestamp() {
    return serverTimestamp;
  }

  @Override
  public int compareTo(JobId o) {
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
    JobId other = (JobId) obj;
    return this.getId() != other.getId();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(64);
    sb.append(JOB_ID_PREFIX).append(serverTimestamp).append("_");
    format(sb, getId());
    return sb.toString();
  }

  private void format(StringBuilder sb, long value) {
    int minimumDigits = JOB_ID_MIN_DIGITS;
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
