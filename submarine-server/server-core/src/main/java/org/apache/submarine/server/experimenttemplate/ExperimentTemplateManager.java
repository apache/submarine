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

package org.apache.submarine.server.experimenttemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.ws.rs.core.Response.Status;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplate;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplateId;
import org.apache.submarine.server.api.spec.ExperimentTemplateSpec;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.experimenttemplate.database.entity.ExperimentTemplateEntity;
import org.apache.submarine.server.experimenttemplate.database.mappers.ExperimentTemplateMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * ExperimentTemplate Management
 */
public class ExperimentTemplateManager {

  private static final Logger LOG =
      LoggerFactory.getLogger(ExperimentTemplateManager.class);

  private static volatile ExperimentTemplateManager manager;

  private final AtomicInteger experimentTemplateIdCounter = new AtomicInteger(0);

  /**
   * ExperimentTemplate Cache
   */
  private final ConcurrentMap<String, ExperimentTemplate> cachedExperimentTemplates =
      new ConcurrentHashMap<>();

  /**
   * Get the singleton instance
   * @return object
   */
  public static ExperimentTemplateManager getInstance() {
    if (manager == null) {
      synchronized (ExperimentTemplateManager.class) {
        if (manager == null) {
          manager = new ExperimentTemplateManager();
        }
      }
    }
    return manager;
  }

  private ExperimentTemplateManager() {

  }

  /**
   * Create ExperimentTemplate
   * @param spec experimentTemplate spec
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate createExperimentTemplate(ExperimentTemplateSpec spec)
      throws SubmarineRuntimeException {
    checkSpec(spec);
    LOG.info("Create ExperimentTemplate using spec: " + spec.toString());
    return createOrUpdateExperimentTemplate(spec, "c");
  }

  /**
   * Update experimentTemplate
   * @param name Name of the experimentTemplate
   * @param spec experimentTemplate spec
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate updateExperimentTemplate(String name, ExperimentTemplateSpec spec)
      throws SubmarineRuntimeException {
    ExperimentTemplate tpl = getExperimentTemplateDetails(name);
    if (tpl == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "ExperimentTemplate not found.");
    }
    checkSpec(spec);
    LOG.info("Update ExperimentTemplate using spec: " + spec.toString());
    return createOrUpdateExperimentTemplate(spec, "u");
  }

  private ExperimentTemplate createOrUpdateExperimentTemplate(ExperimentTemplateSpec spec,
      String operation) {
    ExperimentTemplateEntity entity = new ExperimentTemplateEntity();
    String experimentTemplateId = generateExperimentTemplateId().toString();
    entity.setId(experimentTemplateId);
    entity.setExperimentTemplateName(spec.getName());
    entity.setExperimentTemplateSpec(
        new GsonBuilder().disableHtmlEscaping().create().toJson(spec));
        
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper =
          sqlSession.getMapper(ExperimentTemplateMapper.class);
      if (operation.equals("c")) {
        experimentTemplateMapper.insert(entity);
      } else {
        experimentTemplateMapper.update(entity);
      }
      sqlSession.commit();

      ExperimentTemplate experimentTemplate = new ExperimentTemplate();
      experimentTemplate.setExperimentTemplateId(ExperimentTemplateId.fromString(experimentTemplateId));
      experimentTemplate.setExperimentTemplateSpec(spec);

      // Update cache
      cachedExperimentTemplates.putIfAbsent(spec.getName(), experimentTemplate);

      return experimentTemplate;
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to process the experimentTemplate spec.");
    }
  }

  private ExperimentTemplateId generateExperimentTemplateId() {
    return ExperimentTemplateId.newInstance(SubmarineServer.getServerTimeStamp(),
        experimentTemplateIdCounter.incrementAndGet());
  }

  /**
   * Delete experimentTemplate
   * @param name Name of the experimentTemplate
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate deleteExperimentTemplate(String name)
      throws SubmarineRuntimeException {
    ExperimentTemplate tpl = getExperimentTemplateDetails(name);
    if (tpl == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "ExperimentTemplate not found.");
    }

    LOG.info("Delete ExperimentTemplate for " + name);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper =
          sqlSession.getMapper(ExperimentTemplateMapper.class);
      experimentTemplateMapper.delete(name);
      sqlSession.commit();

      // Invalidate cache
      cachedExperimentTemplates.remove(name);
      return tpl;
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to delete the experimentTemplate.");
    }
  }

  /**
   * Get ExperimentTemplate
   * @param name Name of the experimentTemplate
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate getExperimentTemplate(String name)
      throws SubmarineRuntimeException {
    ExperimentTemplate experimentTemplate = getExperimentTemplateDetails(name);
    if (experimentTemplate == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "ExperimentTemplate not found.");
    }
    return experimentTemplate;
  }

  /**
   * List experimentTemplates
   * @param status experimentTemplate status, if null will return all status
   * @return experimentTemplate list
   * @throws SubmarineRuntimeException the service error
   */
  public List<ExperimentTemplate> listExperimentTemplates(String status)
      throws SubmarineRuntimeException {
    List<ExperimentTemplate> tpls = new ArrayList<>(cachedExperimentTemplates.values());

    // Is it available in cache?
    if (tpls != null && tpls.size() != 0) {
      return tpls;
    }
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper = 
            sqlSession.getMapper(ExperimentTemplateMapper.class);
      List<ExperimentTemplateEntity> experimentTemplateEntitys = experimentTemplateMapper.selectByKey(null);
      for (ExperimentTemplateEntity experimentTemplateEntity : experimentTemplateEntitys) {
        if (experimentTemplateEntity != null) {
          ExperimentTemplate tpl = new ExperimentTemplate();

          tpl.setExperimentTemplateSpec(new Gson().fromJson(
              experimentTemplateEntity.getExperimentTemplateSpec(), ExperimentTemplateSpec.class));
          tpls.add(tpl);
        }
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to get the experimentTemplate details.");
    }
    return tpls;
  }

  private void checkSpec(ExperimentTemplateSpec spec)
      throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Invalid experimentTemplate spec.");
    }
  }

  private ExperimentTemplate getExperimentTemplateDetails(String name)
      throws SubmarineRuntimeException {

    // Is it available in cache?
    ExperimentTemplate tpl = cachedExperimentTemplates.get(name);
    if (tpl != null) {
      return tpl;
    }

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper = 
            sqlSession.getMapper(ExperimentTemplateMapper.class);
      ExperimentTemplateEntity experimentTemplateEntity = experimentTemplateMapper.select(name);

      if (experimentTemplateEntity != null) {
        tpl = new ExperimentTemplate();
        tpl.setExperimentTemplateSpec(new Gson().fromJson(
            experimentTemplateEntity.getExperimentTemplateSpec(), ExperimentTemplateSpec.class));
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to get the experimentTemplate details.");
    }
    return tpl;
  }
}
