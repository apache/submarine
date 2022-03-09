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

package org.apache.submarine.server.database.notebook.service;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.ArrayList;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.notebook.NotebookId;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.database.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.notebook.entity.NotebookEntity;
import org.apache.submarine.server.database.notebook.mappers.NotebookMapper;
import org.joda.time.DateTime;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class NotebookService {

  private static final Logger LOG = LoggerFactory.getLogger(NotebookService.class);

  public List<Notebook> selectAll() throws SubmarineRuntimeException {
    LOG.info("Notebook selectAll");
    List<NotebookEntity> entities;
    List<Notebook> notebooks = new ArrayList<>();
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      entities = mapper.selectAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get notebook entities from database");
    }
    for (NotebookEntity entity : entities) {
      notebooks.add(buildNotebookFromEntity(entity));
    }
    return notebooks;
  }

  public Notebook select(String id) throws SubmarineRuntimeException {
    LOG.info("Notebook select " + id);
    NotebookEntity entity;
    Notebook notebook;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      entity = mapper.select(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get notebook entity from database");
    }
    if (entity != null) {
      notebook = buildNotebookFromEntity(entity);
      return notebook;
    }
    return null;
  }

  public boolean insert(Notebook notebook) throws SubmarineRuntimeException {
    LOG.info("Notebook insert");
    LOG.debug(notebook.toString());
    NotebookEntity entity = buildEntityFromNotebook(notebook);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      mapper.insert(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert notebook entity to database");
    }
    return true;
  }

  public boolean update(Notebook notebook) throws SubmarineRuntimeException {
    LOG.info("Notebook update");
    NotebookEntity entity = buildEntityFromNotebook(notebook);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      mapper.update(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to update notebook entity in database");
    }
    return true;
  }

  public boolean delete(String id) throws SubmarineRuntimeException {
    LOG.info("Notebook delete " + id);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      mapper.delete(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete notebook entity from database");
    }
    return true;
  }

  /**
   * Create a NotebookEntity instance from experiment.
   *
   * @param notebook the Notebook used to create a NoteBookEntity
   * @return NotebookEntity
   */
  private NotebookEntity buildEntityFromNotebook(Notebook notebook) {
    NotebookEntity entity = new NotebookEntity();
    try {
      entity.setId(notebook.getNotebookId().toString());
      entity.setNotebookSpec(new GsonBuilder().disableHtmlEscaping().create().toJson(notebook.getSpec()));
      entity.setNotebookStatus(notebook.getStatus());
      entity.setNotebookUrl(notebook.getUrl());
      entity.setReason(notebook.getReason());
      if (notebook.getCreatedTime() != null) {
        entity.setCreateTime(DateTime.parse(notebook.getCreatedTime()).toDate());
      }
      if (notebook.getDeletedTime() != null) {
        entity.setDeletedTime(DateTime.parse(notebook.getDeletedTime()).toDate());
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to build entity from notebook");
    }
    return entity;
  }

  /**
   * Create a new notebook instance from entity.
   *
   * @param entity the NotebookEntity used to create a Notebook
   * @return Notebook
   */
  private Notebook buildNotebookFromEntity(NotebookEntity entity) {
    Notebook notebook = new Notebook();
    try {
      notebook.setNotebookId(NotebookId.fromString(entity.getId()));
      notebook.setSpec(new Gson().fromJson(entity.getNotebookSpec(), NotebookSpec.class));
      notebook.setName(notebook.getSpec().getMeta().getName());
      notebook.setStatus(entity.getNotebookStatus());
      notebook.setCreatedTime(new DateTime(entity.getCreateTime()).toString());
      notebook.setUrl(entity.getNotebookUrl());
      notebook.setReason(entity.getReason());
      if (entity.getDeletedTime() != null) {
        notebook.setDeletedTime(new DateTime(entity.getDeletedTime()).toString());
      }
      
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to build notebook from entity");
    }
    return notebook;
  }
}
