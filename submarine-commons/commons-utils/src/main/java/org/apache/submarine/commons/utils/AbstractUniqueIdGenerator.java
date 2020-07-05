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

package org.apache.submarine.commons.utils;

public abstract class AbstractUniqueIdGenerator<T> implements Comparable<T> {
  
  private static final int ID_MIN_DIGITS = 4;
  
  private int id;

  private long serverTimestamp;  
  
  /**
   * Get the count since the server started
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
  
  @SuppressWarnings("rawtypes")
  @Override
  public int compareTo(Object o) {
    AbstractUniqueIdGenerator other = (AbstractUniqueIdGenerator) o;
    return this.getId() > other.getId() ? 1 : 0;
  }
  
  @Override
  public int hashCode() {
    final int prime = 371237;
    int result = 6521;
    result = prime * result + (int) (getServerTimestamp() ^ (getServerTimestamp() >>> 32));
    result = prime * result + getId();
    return result;
  }

  @SuppressWarnings("rawtypes")
  @Override
  public boolean equals(Object obj) {
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    if (this == obj) {
      return true;
    }
    AbstractUniqueIdGenerator other = (AbstractUniqueIdGenerator) obj;
    if (this.getServerTimestamp() != other.getServerTimestamp()) {
      return false;
    }
    return this.getId() == other.getId();
  }


  protected void format(StringBuilder sb, long value) {
    int minimumDigits = ID_MIN_DIGITS;
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
