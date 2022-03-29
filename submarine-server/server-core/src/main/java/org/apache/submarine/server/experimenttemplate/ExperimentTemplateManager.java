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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplate;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplateId;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplateSubmit;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;
import org.apache.submarine.server.api.spec.ExperimentTemplateParamSpec;
import org.apache.submarine.server.api.spec.ExperimentTemplateSpec;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.experiment.ExperimentManager;
import org.apache.submarine.server.experimenttemplate.database.entity.ExperimentTemplateEntity;
import org.apache.submarine.server.experimenttemplate.database.mappers.ExperimentTemplateMapper;
import org.apache.submarine.server.utils.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.utils.gson.ExperimentIdSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.minlog.Log;
import com.github.wnameless.json.flattener.JsonFlattener;
import com.github.wnameless.json.unflattener.JsonUnflattener;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * ExperimentTemplate Management.
 */
public class ExperimentTemplateManager {

  private static final Logger LOG = LoggerFactory.getLogger(ExperimentTemplateManager.class);
  private static volatile ExperimentTemplateManager manager;
  private final AtomicInteger experimentTemplateIdCounter = new AtomicInteger(0);

  private static final GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static final Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();



  /**
   * ExperimentTemplate Cache.
   */
  private final ConcurrentMap<String, ExperimentTemplate> cachedExperimentTemplates =
        new ConcurrentHashMap<>();

  /**
   * Get the singleton instance.
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
   * Create ExperimentTemplate.
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
   * Update experimentTemplate.
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

    spec = addResourcesParameter(spec);

    ExperimentTemplateEntity entity = new ExperimentTemplateEntity();
    String experimentTemplateId = generateExperimentTemplateId().toString();
    entity.setId(experimentTemplateId);
    entity.setExperimentTemplateName(spec.getName());
    entity.setExperimentTemplateSpec(gsonBuilder.disableHtmlEscaping().create().toJson(spec));


    parameterMapping(spec);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper =
            sqlSession.getMapper(ExperimentTemplateMapper.class);

      if (operation.equals("c")) {
        experimentTemplateMapper.insert(entity);
      } else {
        experimentTemplateMapper.update(entity);
      }
      sqlSession.commit();

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to insert or update the experimentTemplate spec: " + e.getMessage());
    }

    ExperimentTemplate experimentTemplate;
    try {
      experimentTemplate = new ExperimentTemplate();
      experimentTemplate.setExperimentTemplateId(ExperimentTemplateId.fromString(experimentTemplateId));
      experimentTemplate.setExperimentTemplateSpec(spec);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to parse the experimentTemplate spec: " + e.getMessage());
    }
    // Update cache
    cachedExperimentTemplates.put(spec.getName(), experimentTemplate);

    return experimentTemplate;
  }

  private ExperimentTemplateId generateExperimentTemplateId() {
    return ExperimentTemplateId.newInstance(SubmarineServer.getServerTimeStamp(),
        experimentTemplateIdCounter.incrementAndGet());
  }

  /**
   * Delete experimentTemplate.
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
   * Get ExperimentTemplate.
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
   * List experimentTemplates.
   *
   * @param status experimentTemplate status, if null will return all status
   * @return experimentTemplate list
   * @throws SubmarineRuntimeException the service error
   */
  public List<ExperimentTemplate> listExperimentTemplates(String status) throws SubmarineRuntimeException {
    List<ExperimentTemplate> tpls = new ArrayList<>(cachedExperimentTemplates.values());

    // Is it available in cache?
    if (tpls.size() != 0) {
      return tpls;
    }
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentTemplateMapper experimentTemplateMapper =
            sqlSession.getMapper(ExperimentTemplateMapper.class);

      List<ExperimentTemplateEntity> experimentTemplateEntities = experimentTemplateMapper.selectByKey(null);
      for (ExperimentTemplateEntity experimentTemplateEntity : experimentTemplateEntities) {
        if (experimentTemplateEntity != null) {
          ExperimentTemplate tpl = new ExperimentTemplate();

          tpl.setExperimentTemplateSpec(
                gson.fromJson(experimentTemplateEntity.getExperimentTemplateSpec(),
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
            gson.fromJson(experimentTemplateEntity.getExperimentTemplateSpec(),
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
   * Create ExperimentTemplate.
   *
   * @param submittedParam experimentTemplate spec
   * @return Experiment experiment
   * @throws SubmarineRuntimeException the service error
   */
  public Experiment submitExperimentTemplate(ExperimentTemplateSubmit submittedParam)
        throws SubmarineRuntimeException {

    if (submittedParam == null) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
            "Invalid ExperimentTemplateSubmit spec.");
    }

    ExperimentTemplate experimentTemplate = getExperimentTemplate(submittedParam.getName());
    Map<String, String> params = submittedParam.getParams();


    for (ExperimentTemplateParamSpec paramSpec:
          experimentTemplate.getExperimentTemplateSpec().getExperimentTemplateParamSpec()) {

      String value = params.get(paramSpec.getName());
      if (value != null) {
        paramSpec.setValue(value);
      }
    }
    ExperimentTemplateSpec spec = experimentTemplate.getExperimentTemplateSpec();

    ExperimentSpec experimentSpec = parameterMapping(spec, params);

    return ExperimentManager.getInstance().createExperiment(experimentSpec);
  }


  private ExperimentTemplateSpec addResourcesParameter(ExperimentTemplateSpec tplSpec) {

    for (Map.Entry<String, ExperimentTaskSpec> entrySet : tplSpec.getExperimentSpec().getSpec().entrySet()) {

      ExperimentTaskSpec taskSpec = entrySet.getValue();
      // parse resourceMap
      taskSpec.setResources(taskSpec.getResources());

      ExperimentTemplateParamSpec parm1 = new ExperimentTemplateParamSpec();
      parm1.setName(String.format("spec.%s.replicas", entrySet.getKey()));
      parm1.setValue(taskSpec.getReplicas() == null ? "1" : taskSpec.getReplicas().toString());
      parm1.setRequired("false");
      parm1.setDescription("");
      tplSpec.getExperimentTemplateParamSpec().add(parm1);

      ExperimentTemplateParamSpec parm2 = new ExperimentTemplateParamSpec();
      parm2.setName(String.format("spec.%s.resourceMap.cpu", entrySet.getKey()));
      parm2.setValue(taskSpec.getCpu() == null ? "1" : taskSpec.getCpu());
      parm2.setRequired("false");
      parm2.setDescription("");
      tplSpec.getExperimentTemplateParamSpec().add(parm2);

      ExperimentTemplateParamSpec parm3 = new ExperimentTemplateParamSpec();
      parm3.setName(String.format("spec.%s.resourceMap.memory", entrySet.getKey()));
      parm3.setValue(taskSpec.getMemory() == null ? "1" : taskSpec.getMemory().toString());
      parm3.setRequired("false");
      parm3.setDescription("");
      tplSpec.getExperimentTemplateParamSpec().add(parm3);

    }
    return tplSpec;
  }


  // use template default value to mapping
  private ExperimentSpec parameterMapping(ExperimentTemplateSpec tplSpec) {

    Map<String, String> paramMap = new HashMap<String, String>();
    for (ExperimentTemplateParamSpec parm : tplSpec.getExperimentTemplateParamSpec()) {
      if (parm.getValue() != null) {
        paramMap.put(parm.getName(), parm.getValue());
      } else {
        paramMap.put(parm.getName(), "");
      }
    }
    return parameterMapping(tplSpec, paramMap);
  }

  private ExperimentSpec parameterMapping(ExperimentTemplateSpec tplspec, Map<String, String> paramMap) {

    String spec = gson.toJson(tplspec.getExperimentSpec());
    Map<String, Object> flattenJson = JsonFlattener.flattenAsMap(spec);

    Log.info(flattenJson.toString());
    // illegalParamList: The parameters not in template parameters should not be used
    // Check at submission
    Map<String, ExperimentTemplateParamSpec> tplparamMap = new HashMap<String, ExperimentTemplateParamSpec>();
    for (ExperimentTemplateParamSpec tplParam : tplspec.getExperimentTemplateParamSpec()) {
      tplparamMap.put(tplParam.getName(), tplParam);
    }
    Set<String> illegalParamList = new HashSet<String>();
    for (String key : paramMap.keySet()) {
      if (tplparamMap.get(key) == null) {
        illegalParamList.add(key);
      }
    }

    if (illegalParamList.size() > 0) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
            "Parameters contains illegal key: " + illegalParamList.toString());
    }

    // unmapedParamList: Parameters that should be used in the template but could not be found
    // Check at registration and submission
    Set<String> unmapedParamList = new HashSet<String>();
    for (ExperimentTemplateParamSpec tplParam : tplspec.getExperimentTemplateParamSpec()) {
      if (paramMap.get(tplParam.getName()) == null) {
        // use default value
        if (!Boolean.parseBoolean(tplParam.getRequired())) {
          paramMap.put(tplParam.getName(), tplParam.getValue());
        } else {
          unmapedParamList.add(tplParam.getName());
        }
      }
    }

    // unusedParamList: Parameters not used by the template
    // Check at registration
    Set<String> unusedParamList = new HashSet<String>();
    for (String s : paramMap.keySet()) {
      unusedParamList.add(s);
    }

    // resourceMap needs special handling
    for (Map.Entry<String, ExperimentTaskSpec> entrySet : tplspec.getExperimentSpec().getSpec().entrySet()) {
      String cpu = paramMap.get(String.format("spec.%s.resourceMap.cpu", entrySet.getKey()));
      String memory = paramMap.get(String.format("spec.%s.resourceMap.memory", entrySet.getKey()));
      flattenJson.put(String.format("spec.%s.resources", entrySet.getKey()),
            String.format("cpu=%s,memory=%s", cpu, memory));
    }

    // Mapping the {{...}} param
    Pattern pattern = Pattern.compile("\\{\\{(.*?)\\}\\}");
    for (Map.Entry<String, Object> entry : flattenJson.entrySet()) {
      boolean isMatch = false;
      if (entry.getValue() instanceof String) {
        String value = (String) entry.getValue();
        Matcher matcher = pattern.matcher(value);
        StringBuffer sb = new StringBuffer();

        while (matcher.find()) {
          String name = matcher.group(1);
          String key = entry.getKey() + ":" + name;

          // match path+placeholder  ("meta.cmd:parametername")
          if (paramMap.get(key) != null) {
            isMatch = true;
            matcher.appendReplacement(sb, paramMap.get(key));
            unusedParamList.remove(key);
            unmapedParamList.remove(key);
          }
          // match placeholder ("parametername")
          else if (paramMap.get(name) != null) {
            isMatch = true;
            matcher.appendReplacement(sb, paramMap.get(name));
            unusedParamList.remove(name);
            unmapedParamList.remove(name);
          } else {
            unmapedParamList.add(key);
          }
        }
        if (isMatch) {
          matcher.appendTail(sb);
          flattenJson.put(entry.getKey(), sb.toString());
        }
      }
      // match path ("meta.cmd")
      if (!isMatch) {
        String key = entry.getKey();
        if (paramMap.get(key) != null) {
          flattenJson.put(key, paramMap.get(key));
          unusedParamList.remove(key);
        }
      }
    }

    if (unusedParamList.size() > 0) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
            "Parameters contains unused key: " + unusedParamList.toString());
    }

    if (unmapedParamList.size() > 0) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Template contains unmapped value: " + unmapedParamList.toString());
    }

    String json = flattenJson.toString();
    Log.info("flattenJson    " + json);

    String nestedJson = JsonUnflattener.unflatten(json);
    Log.info("nestedJson    " + nestedJson);

    ExperimentSpec returnExperimentSpec = null;
    try {
      returnExperimentSpec = gson.fromJson(nestedJson, ExperimentSpec.class);
      Log.info("ExperimentSpec " + returnExperimentSpec.toString());

    } catch (Exception e) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Template mapping fail: " + e.getMessage() + nestedJson);
    }
    return returnExperimentSpec;
  }
}
