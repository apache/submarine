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

package org.apache.submarine.server.experiment;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.ws.rs.core.Response.Status;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.SubmitterManager;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.experiment.database.ExperimentEntity;
import org.apache.submarine.server.experiment.database.ExperimentService;
import org.apache.submarine.server.rest.RestConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * It's responsible for managing the experiment CRUD and cache them
 */
public class ExperimentManager {
  private static final Logger LOG = LoggerFactory.getLogger(ExperimentManager.class);

  private static volatile ExperimentManager manager;

  private final AtomicInteger experimentCounter = new AtomicInteger(0);

  /**
   * Used to cache the specs by the experiment id.
   * key: the string of experiment id
   * value: Experiment object
   */
  private final ConcurrentMap<String, Experiment> cachedExperimentMap = new ConcurrentHashMap<>();

  private final Submitter submitter;
  private final ExperimentService experimentService = new ExperimentService();

  /**
   * Get the singleton instance
   *
   * @return object
   */
  public static ExperimentManager getInstance() {
    if (manager == null) {
      synchronized (ExperimentManager.class) {
        if (manager == null) {
          manager = new ExperimentManager(SubmitterManager.loadSubmitter());
        }
      }
    }
    return manager;
  }

  private ExperimentManager(Submitter submitter) {
    this.submitter = submitter;
  }

  /**
   * Create experiment
   *
   * @param spec spec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Experiment createExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    checkSpec(spec);

    // Submarine sdk will get experimentID and JDBC URL from environment variables in each worker,
    // and then log experiment metrics and parameters to submarine server
    ExperimentId id = generateExperimentId();
    String url = getSQLAlchemyURL();
    spec.getMeta().getEnvVars().put(RestConstants.JOB_ID, id.toString());
    spec.getMeta().getEnvVars().put(RestConstants.SUBMARINE_TRACKING_URI, url);
    String lowerName = spec.getMeta().getName().toLowerCase();
    spec.getMeta().setName(lowerName);

    Experiment experiment = submitter.createExperiment(spec);
    experiment.setExperimentId(id);

    spec.getMeta().getEnvVars().remove(RestConstants.JOB_ID);
    spec.getMeta().getEnvVars().remove(RestConstants.SUBMARINE_TRACKING_URI);
    experiment.setSpec(spec);

    ExperimentEntity entity = new ExperimentEntity();
    entity.setId(experiment.getExperimentId().toString());
    entity.setExperimentSpec(new GsonBuilder().disableHtmlEscaping().create().toJson(experiment.getSpec()));
    experimentService.insert(entity);

    return experiment;
  }

  /**
   * Get experiment
   *
   * @param id experiment id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Experiment getExperiment(String id) throws SubmarineRuntimeException {
    checkExperimentId(id);

    ExperimentEntity entity = experimentService.select(id);
    ExperimentSpec spec = new Gson().fromJson(entity.getExperimentSpec(), ExperimentSpec.class);
    Experiment experiment = submitter.findExperiment(spec);

    return experiment;
  }

  /**
   * List experiments
   *
   * @param status status, if null will return all experiments
   * @return list
   * @throws SubmarineRuntimeException the service error
   */
  public List<Experiment> listExperimentsByStatus(String status) throws SubmarineRuntimeException {
    List<Experiment> experimentList = new ArrayList<>();
    List<ExperimentEntity> entities = experimentService.selectAll();

    for (ExperimentEntity entity: entities) {
      ExperimentSpec spec = new Gson().fromJson(entity.getExperimentSpec(), ExperimentSpec.class);
      Experiment experiment = submitter.findExperiment(spec);
      LOG.info("Found experiment: {}", experiment.getStatus());
      if (status == null || status.toLowerCase().equals(experiment.getStatus().toLowerCase())) {
        experimentList.add(experiment);
      }
    }
    LOG.info("List experiment: {}", experimentList.size());
    return experimentList;
  }

  /**
   * Patch the experiment
   *
   * @param id   experiment id
   * @param spec spec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Experiment patchExperiment(String id, ExperimentSpec spec) throws SubmarineRuntimeException {
    checkExperimentId(id);
    checkSpec(spec);

    ExperimentEntity entity = new ExperimentEntity();
    entity.setId(id);
    entity.setExperimentSpec(new GsonBuilder().disableHtmlEscaping().create().toJson(spec));

    Experiment experiment = submitter.patchExperiment(spec);

    // updated spec
    entity.setExperimentSpec(new GsonBuilder().disableHtmlEscaping().create().toJson(experiment.getSpec()));
    experimentService.update(entity);

    return experiment;
  }

  /**
   * Delete experiment
   *
   * @param id experiment id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Experiment deleteExperiment(String id) throws SubmarineRuntimeException {
    checkExperimentId(id);

    ExperimentEntity entity = experimentService.select(id);
    ExperimentSpec spec = new Gson().fromJson(entity.getExperimentSpec(), ExperimentSpec.class);

    Experiment experiment = submitter.deleteExperiment(spec);
    experimentService.delete(id);

    return experiment;
  }

  /**
   * List experiment logs
   *
   * @param status status, if null will return all experiment logs
   * @return log list
   * @throws SubmarineRuntimeException the service error
   */
  public List<ExperimentLog> listExperimentLogsByStatus(String status) throws SubmarineRuntimeException {
    List<ExperimentLog> experimentLogList = new ArrayList<>();
    List<ExperimentEntity> entities = experimentService.selectAll();

    for (ExperimentEntity entity: entities) {
      String id = entity.getId();
      ExperimentSpec spec = new Gson().fromJson(entity.getExperimentSpec(), ExperimentSpec.class);
      Experiment experiment = submitter.findExperiment(spec);

      LOG.info("Found experiment: {}", experiment.getStatus());

      if (status == null || status.toLowerCase().equals(experiment.getStatus().toLowerCase())) {
        experimentLogList.add(submitter.getExperimentLogName(spec, id));
      }

    }
    return experimentLogList;
  }

  /**
   * Get experiment log
   *
   * @param id experiment id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentLog getExperimentLog(String id) throws SubmarineRuntimeException {
    checkExperimentId(id);

    ExperimentEntity entity = experimentService.select(id);
    ExperimentSpec spec = new Gson().fromJson(entity.getExperimentSpec(), ExperimentSpec.class);
    Experiment experiment = submitter.findExperiment(spec);

    return submitter.getExperimentLog(spec, id);
  }

  private void checkSpec(ExperimentSpec spec) throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.OK.getStatusCode(), "Invalid experiment spec.");
    }
  }

  private void checkExperimentId(String id) throws SubmarineRuntimeException {
    ExperimentId experimentId = ExperimentId.fromString(id);
    if (experimentId == null || !cachedExperimentMap.containsKey(id)) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(), "Not found experiment.");
    }
  }

  private String getSQLAlchemyURL() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    String jdbcUrl = conf.getJdbcUrl();
    jdbcUrl = jdbcUrl.substring(jdbcUrl.indexOf("//") + 2, jdbcUrl.indexOf("?"));
    String jdbcUserName = conf.getJdbcUserName();
    String jdbcPassword = conf.getJdbcPassword();
    return "mysql+pymysql://" + jdbcUserName + ":" + jdbcPassword + "@" + jdbcUrl;
  }

  private ExperimentId generateExperimentId() {
    return ExperimentId.newInstance(SubmarineServer.getServerTimeStamp(),
      experimentCounter.incrementAndGet());
  }
}
