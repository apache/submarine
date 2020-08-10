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

package org.apache.submarine.server.api.notebook;

import org.apache.submarine.commons.utils.AbstractUniqueIdGenerator;

/**
 * The unique id for notebook. Formatter:
 * notebook_${server_timestamp}_${counter} Such as:
 * notebook_1577627710_0001
 */
public class NotebookId extends AbstractUniqueIdGenerator<NotebookId> {
  private static final String NOTEBOOK_ID_PREFIX = "notebook_";

  /**
   * Get the object of NotebookId.
   * @param notebookId string
   * @return object
   */
  public static NotebookId fromString(String notebookId) {
    if (notebookId == null) {
      return null;
    }
    String[] components = notebookId.split("\\_");
    if (components.length != 3) {
      return null;
    }
    return NotebookId.newInstance(Long.parseLong(components[1]), Integer.parseInt(components[2]));
  }

  /**
   * Get the object of NotebookId.
   * @param serverTimestamp timestamp
   * @param id count
   * @return object
   */
  public static NotebookId newInstance(long serverTimestamp, int id) {
    NotebookId notebookId = new NotebookId();
    notebookId.setServerTimestamp(serverTimestamp);
    notebookId.setId(id);
    return notebookId;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(64);
    sb.append(NOTEBOOK_ID_PREFIX).append(getServerTimestamp()).append("_");
    format(sb, getId());
    return sb.toString();
  }

}
