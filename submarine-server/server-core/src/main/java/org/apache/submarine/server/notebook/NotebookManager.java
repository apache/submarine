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

package org.apache.submarine.server.notebook;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.SubmitterManager;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.notebook.NotebookId;
import org.apache.submarine.server.api.spec.NotebookSpec;

import javax.ws.rs.core.Response;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

public class NotebookManager {

  private static volatile NotebookManager manager;

  private final Submitter submitter;

  private NotebookManager(Submitter submitter) {
    this.submitter = submitter;
  }

  private final AtomicInteger notebookCounter = new AtomicInteger(0);

  /**
   * Used to cache the specs by the notebook id.
   *  key: the string of notebook id
   *  value: Notebook object
   */
  private final ConcurrentMap<String, Notebook> cachedNotebookMap = new ConcurrentHashMap<>();

  /**
   * Get the singleton instance
   * @return object
   */
  public static NotebookManager getInstance() {
    if (manager == null) {
      synchronized (NotebookManager.class) {
        if (manager == null) {
          manager = new NotebookManager(SubmitterManager.loadSubmitter());
        }
      }
    }
    return manager;
  }

  /**
   * Create a notebook instance
   * @param spec NotebookSpec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Notebook createNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    checkNotebookSpec(spec);
    Notebook notebook = submitter.createNotebook(spec);
    notebook.setNotebookId(generateNotebookId());
    notebook.setSpec(spec);
    cachedNotebookMap.putIfAbsent(notebook.getNotebookId().toString(), notebook);
    return notebook;
  }

  /**
   * List notebook instances
   * @param status status, if null will return all notebooks
   * @return list
   * @throws SubmarineRuntimeException the service error
   */
  public List<Notebook> listNotebooksByStatus(String status) throws SubmarineRuntimeException {
    //TODO(ryan): implement the method
    return null;
  }

  /**
   * Get a notebook instance
   * @param id notebook id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Notebook getNotebook(String id) throws SubmarineRuntimeException {
    //TODO(ryan): implement the method
    return null;
  }

  /**
   * Delete the notebook instance
   * @param id notebook id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Notebook deleteNotebook(String id) throws SubmarineRuntimeException {
    //TODO(ryan): implement the method
    return null;
  }

  /**
   * Generate a unique notebook id
   * @return notebook id
   */
  private NotebookId generateNotebookId() {
    return NotebookId.newInstance(SubmarineServer.getServerTimeStamp(),
            notebookCounter.incrementAndGet());
  }

  /**
   * Check if notebook spec is valid
   * @param spec notebook spec
   */
  private void checkNotebookSpec(NotebookSpec spec) {
    //TODO(ryan): The method need to be improved
    if (spec == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
              "Invalid. Notebook Spec object is null.");
    }
  }

}
