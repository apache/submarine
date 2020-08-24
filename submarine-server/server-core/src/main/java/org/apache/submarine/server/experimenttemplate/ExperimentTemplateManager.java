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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.ws.rs.core.Response.Status;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplate;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplateId;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplateSubmit;
import org.apache.submarine.server.api.spec.ExperimentTemplateParamSpec;
import org.apache.submarine.server.api.spec.ExperimentTemplateSpec;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.experiment.ExperimentManager;
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

  private static final Logger LOG = LoggerFactory.getLogger(ExperimentTemplateManager.class);

  private static volatile ExperimentTemplateManager manager;

  private final AtomicInteger experimentTemplateIdCounter = new AtomicInteger(0);

  /**
   * ExperimentTemplate Cache
   */
  private final ConcurrentMap<String, ExperimentTemplate> cachedExperimentTemplates = 
        new ConcurrentHashMap<>();

  /**
   * Get the singleton instance
   * 
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
   * 
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
   * 
   * @param name Name of the experimentTemplate
   * @param spec experimentTemplate spec
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate updateExperimentTemplate(String name, ExperimentTemplateSpec spec)
      throws SubmarineRuntimeException {
    ExperimentTemplate tpl = getExperimentTemplateDetails(name);
    if (tpl == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(), "ExperimentTemplate not found.");
    }
    checkSpec(spec);
    LOG.info("Update ExperimentTemplate using spec: " + spec.toString());
    return createOrUpdateExperimentTemplate(spec, "u");
  }

  private ExperimentTemplate createOrUpdateExperimentTemplate(ExperimentTemplateSpec spec, String operation) {
    ExperimentTemplateEntity entity = new ExperimentTemplateEntity();
    String experimentTemplateId = generateExperimentTemplateId().toString();
    entity.setId(experimentTemplateId);
    entity.setExperimentTemplateName(spec.getName());
    entity.setExperimentTemplateSpec(new GsonBuilder().disableHtmlEscaping().create().toJson(spec));

    parameterMapping(entity.getExperimentTemplateSpec());

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
   * 
   * @param name Name of the experimentTemplate
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate deleteExperimentTemplate(String name) throws SubmarineRuntimeException {
    ExperimentTemplate tpl = getExperimentTemplateDetails(name);
    if (tpl == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(), "ExperimentTemplate not found.");
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
   * 
   * @param name Name of the experimentTemplate
   * @return ExperimentTemplate experimentTemplate
   * @throws SubmarineRuntimeException the service error
   */
  public ExperimentTemplate getExperimentTemplate(String name) throws SubmarineRuntimeException {
    ExperimentTemplate experimentTemplate = getExperimentTemplateDetails(name);
    if (experimentTemplate == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(), "ExperimentTemplate not found.");
    }
    return experimentTemplate;
  }

  /**
   * List experimentTemplates
   * 
   * @param status experimentTemplate status, if null will return all status
   * @return experimentTemplate list
   * @throws SubmarineRuntimeException the service error
   */
  public List<ExperimentTemplate> listExperimentTemplates(String status) throws SubmarineRuntimeException {
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

          tpl.setExperimentTemplateSpec(
                new Gson().fromJson(experimentTemplateEntity.getExperimentTemplateSpec(), 
                ExperimentTemplateSpec.class));
          tpls.add(tpl);
          cachedExperimentTemplates.put(tpl.getExperimentTemplateSpec().getName(), tpl);
        }
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to get the experimentTemplate details.");
    }
    return tpls;
  }

  private void checkSpec(ExperimentTemplateSpec spec) throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(), 
            "Invalid experimentTemplate spec.");
    }
  }

  private ExperimentTemplate getExperimentTemplateDetails(String name) throws SubmarineRuntimeException {

    // Is it available in cache?
    ExperimentTemplate tpl = cachedExperimentTemplates.get(name);
    if (tpl != null) {
      return tpl;
    }
    ExperimentTemplateEntity experimentTemplateEntity;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper = 
            sqlSession.getMapper(ExperimentTemplateMapper.class);

      experimentTemplateEntity = experimentTemplateMapper.select(name);

      if (experimentTemplateEntity != null) {
        tpl = new ExperimentTemplate();
        tpl.setExperimentTemplateSpec(
            new Gson().fromJson(experimentTemplateEntity.getExperimentTemplateSpec(), 
            ExperimentTemplateSpec.class));
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to get the experimentTemplate details.");
    }
    return tpl;
  }


  /**
   * Create ExperimentTemplate
   * 
   * @param SubmittedParam experimentTemplate spec
   * @return Experiment experiment
   * @throws SubmarineRuntimeException the service error
   */
  public Experiment submitExperimentTemplate(ExperimentTemplateSubmit SubmittedParam) 
        throws SubmarineRuntimeException {

    if (SubmittedParam == null) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(), 
            "Invalid ExperimentTemplateSubmit spec.");
    }

    ExperimentTemplate experimentTemplate = getExperimentTemplate(SubmittedParam.getName());
    Map<String, String> params = SubmittedParam.getParams();


    for (ExperimentTemplateParamSpec paramSpec: experimentTemplate.getExperimentTemplateSpec().getParameters()) {

      String value = sparam.get(tpaam.getName());
      if (value != null) {
        tpaam.setValue(value);
      }
    }
    String spec = new Gson().toJson(experimentTemplate.getExperimentTemplateSpec());

    ExperimentTemplateSpec experimentTemplateSpec = parameterMapping(spec, sparam);
        
    return ExperimentManager.getInstance().createExperiment(experimentTemplateSpec.getExperimentSpec());
  }

  private ExperimentTemplateSpec parameterMapping(String spec) {
    ExperimentTemplateSpec tplSpec = new Gson().fromJson(spec, ExperimentTemplateSpec.class);

    Map<String, String> paramMap = new HashMap<String, String>();
    for (ExperimentTemplateParamSpec parm : tplSpec.getParameters()) {
      if (parm.getValue() != null) {
        paramMap.put(parm.getName(), parm.getValue());
      } else {
        paramMap.put(parm.getName(), "");
      }
    }

    return parameterMapping(spec, paramMap);
  }

  // Use params to replace the content of spec
  private ExperimentTemplateSpec parameterMapping(String spec, Map<String, String> paramMap) {

    Pattern pattern = Pattern.compile("\\{\\{(.+?)\\}\\}");
    StringBuffer sb = new StringBuffer();
    Matcher matcher = pattern.matcher(spec);

    List<String> unmappedKeys = new ArrayList<String>();

    while (matcher.find()) {
      final String key = matcher.group(1);
      final String replacement = paramMap.get(key);
      if (replacement == null) {
        unmappedKeys.add(key);
      }
      else {
        matcher.appendReplacement(sb, replacement);
      }
      paramMap.remove(key);
    }
    matcher.appendTail(sb);

    if (paramMap.size() > 0) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
            "Parameters contains unused key: " + paramMap.keySet());
    }

    if (unmappedKeys.size() > 0) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Template contains unmapped key: " + unmappedKeys);
    }  
    ExperimentTemplateSpec tplSpec;
    try {
      tplSpec = new Gson().fromJson(sb.toString(), ExperimentTemplateSpec.class);
    } catch (Exception e) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Template mapping fail: " + e.getMessage() + sb.toString());
    }
    return tplSpec;
  }
}
