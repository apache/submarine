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
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.notebook.NotebookId;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.environment.EnvironmentManager;

import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.submarine.server.notebook.database.service.NotebookService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NotebookManager {
  private static final Logger LOG = LoggerFactory.getLogger(NotebookManager.class);

  private static volatile NotebookManager manager;

  private final Submitter submitter;

  private final NotebookService notebookService;

  private NotebookManager(Submitter submitter, NotebookService notebookService) {
    this.submitter = submitter;
    this.notebookService = notebookService;
  }

  private final AtomicInteger notebookCounter = new AtomicInteger(0);

  /**
   * Get the singleton instance.
   *
   * @return object
   */
  public static NotebookManager getInstance() {
    if (manager == null) {
      synchronized (NotebookManager.class) {
        if (manager == null) {
          manager = new NotebookManager(SubmitterManager.loadSubmitter(), new NotebookService());
        }
      }
    }
    return manager;
  }

  /**
   * Create a notebook instance.
   *
   * @param spec NotebookSpec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Notebook createNotebook(NotebookSpec spec) throws SubmarineRuntimeException {
    checkNotebookSpec(spec);
    String lowerName = spec.getMeta().getName().toLowerCase();
    spec.getMeta().setName(lowerName);
    NotebookId notebookId = generateNotebookId();

    Map<String, String> labels = spec.getMeta().getLabels();

    if (labels == null) {
      labels = new HashMap<>();
    }
    labels.put("notebook-owner-id", spec.getMeta().getOwnerId());
    labels.put("notebook-id", notebookId.toString());
    spec.getMeta().setLabels(labels);
    Notebook notebook = submitter.createNotebook(spec, notebookId.toString());

    notebook.setNotebookId(notebookId);
    notebook.setSpec(spec);

    // environment information
    NotebookSpec notebookSpec = notebook.getSpec();
    EnvironmentManager environmentManager = EnvironmentManager.getInstance();
    Environment environment = environmentManager.getEnvironment(spec.getEnvironment().getName());
    if (environment.getEnvironmentSpec() != null) {
      notebookSpec.setEnvironment(environment.getEnvironmentSpec());
    }
    notebook.setStatus(Notebook.Status.STATUS_WAITING.getValue());
    notebookService.insert(notebook);
    return notebook;
  }

  /**
   * List notebook instances.
   *
   * @param namespace namespace, if null will return all notebooks
   * @return list
   * @throws SubmarineRuntimeException the service error
   */
  public List<Notebook> listNotebooksByNamespace(String namespace) throws SubmarineRuntimeException {
    List<Notebook> notebookList = new ArrayList<>();
    for (Notebook notebook : notebookService.selectAll()) {
      if (namespace == null || namespace.length() == 0 ){
        if (notebook.getStatus().equals(Notebook.Status.STATUS_CREATING.getValue())) {
          Notebook patchNotebook = submitter.findNotebook(notebook.getSpec());
          notebook.rebuild(patchNotebook);
          notebookList.add(notebook);
        } else {
          notebookList.add(notebook);
        }
      }
    }
    return notebookList;
  }

  /**
   * Get a list of notebook with user id.
   *
   * @param id user id
   * @return a list of notebook
   */
  public List<Notebook> listNotebooksByUserId(String id) {
    List<Notebook> serviceNotebooks = notebookService.selectAll();
    List<Notebook> notebookList = new ArrayList<>();
    for (Notebook nb : serviceNotebooks) {
      try {
        if (nb.getStatus().equals(Notebook.Status.STATUS_CREATING.getValue())) {
          Notebook patchNotebook = submitter.findNotebook(nb.getSpec());
          nb.rebuild(patchNotebook);
          notebookList.add(nb);
        } else {
          notebookList.add(nb);
        }
      } catch (SubmarineRuntimeException e) {
        LOG.error("Error when get notebook resource, skip this row!", e);
      }
    }
    return notebookList;
  }

  /**
   * Get a notebook instance.
   *
   * @param id notebook id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Notebook getNotebook(String id) throws SubmarineRuntimeException {
    checkNotebookId(id);

    Notebook notebook = notebookService.select(id);
    if (notebook == null) {
      throw new SubmarineRuntimeException(Response.Status.NOT_FOUND.getStatusCode(),
          "Notebook not found.");
    }
    Notebook foundNotebook = submitter.findNotebook(notebook.getSpec());
    foundNotebook.rebuild(notebook);
    foundNotebook.setNotebookId(NotebookId.fromString(id));

    return foundNotebook;
  }

  /**
   * Delete the notebook instance.
   *
   * @param id notebook id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Notebook deleteNotebook(String id) throws SubmarineRuntimeException {
    Notebook notebook = getNotebook(id);
    submitter.deleteNotebook(notebook.getSpec());
    notebookService.delete(id);
    return notebook;
  }

  /**
   * Generate a unique notebook id.
   *
   * @return notebook id
   */
  private NotebookId generateNotebookId() {
    return NotebookId.newInstance(SubmarineServer.getServerTimeStamp(),
        notebookCounter.incrementAndGet());
  }

  /**
   * Check if notebook spec is valid.
   *
   * @param spec notebook spec
   */
  private void checkNotebookSpec(NotebookSpec spec) {
    //TODO(ryan): The method need to be improved
    if (spec == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Notebook Spec object is null.");
    }
    List<Notebook> serviceNotebooks = notebookService.selectAll();
    for (Notebook notebook: serviceNotebooks) {
      if (notebook.getSpec().getMeta().getName().equals(spec.getMeta().getName())) {
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Notebook with same name is already existed.");
      }
    }
  }

  private void checkNotebookId(String id) throws SubmarineRuntimeException {
    NotebookId notebookId = NotebookId.fromString(id);
    if (notebookId == null) {
      throw new SubmarineRuntimeException(Response.Status.NOT_FOUND.getStatusCode(),
          "Notebook not found.");
    }
  }

}
