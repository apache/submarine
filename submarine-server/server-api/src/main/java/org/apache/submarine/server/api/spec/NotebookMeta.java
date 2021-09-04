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

package org.apache.submarine.server.api.spec;

import java.util.Map;

public class NotebookMeta {
  private String name;
  private String namespace;
  private String ownerId;
  private Map<String, String> labels;

  public NotebookMeta() {

  }

  /**
   * Get the notebook name which is unique within a namespace.
   * @return notebook name
   */
  public String getName() {
    return name;
  }

  /**
   * Name must be unique within a namespace. Is required when creating notebook.
   * @param name notebook name
   */
  public void setName(String name) {
    this.name = name;
  }

  /**
   * Get the namespace which defines the isolated space for each notebook.
   * @return namespace
   */
  public String getNamespace() {
    return namespace;
  }

  /**
   * Namespace defines the space within each name must be unique.
   * @param namespace namespace
   */
  public void setNamespace(String namespace) {
    this.namespace = namespace;
  }

  /**
   * Get the ownerId
   * @return ownerId
   */
  public String getOwnerId() {
    return ownerId;
  }

  /**
   * Set the ownerId
   * @param ownerId ownerId
   */
  public void setOwnerId(String ownerId) {
    this.ownerId = ownerId;
  }

  /**
   * Set the labels on Notebook
   * @param Map labels
   */
  public Map<String, String> getLabels() {
    return labels;
  }
  /**
   * get labels on Notebook
   * @return labels
   */
  public void setLabels(Map<String, String> labels) {
    this.labels = labels;
  }

}
