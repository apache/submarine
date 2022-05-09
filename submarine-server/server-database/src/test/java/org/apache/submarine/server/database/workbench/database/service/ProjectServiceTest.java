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
package org.apache.submarine.server.database.workbench.service;

import org.apache.submarine.server.database.workbench.entity.ProjectEntity;
import org.apache.submarine.server.database.workbench.entity.ProjectFilesEntity;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class ProjectServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(ProjectServiceTest.class);
  ProjectService projectService = new ProjectService();

  @After
  public void removeAllRecord() throws Exception {
    List<ProjectEntity> projectList = projectService.queryPageList(null, "create_time", "desc", 0, 100);
    LOG.info("projectList.size():{}", projectList.size());
    for (ProjectEntity project : projectList) {
      projectService.delete(project.getId());
    }
  }

  @Test
  public void queryPageList() throws Exception {
    ProjectFilesEntity projectFiles = new ProjectFilesEntity();
    projectFiles.setFileContent("ProjectServiceTest-FileContent");
    projectFiles.setFileName("ProjectServiceTest-FileName");
    projectFiles.setCreateBy("ProjectServiceTest-UserName");

    ProjectEntity project = new ProjectEntity();
    project.setDescription("ProjectServiceTest-Description");
    project.setName("ProjectServiceTest-ProjectName");
    project.setType("PROJECT_TYPE_NOTEBOOK");
    project.setUserName("ProjectServiceTest-UserName");
    project.setVisibility("PROJECT_VISIBILITY_PRIVATE");
    project.setCreateBy("ProjectServiceTest-UserName");
    List list = new ArrayList<ProjectFilesEntity>();
    list.add(projectFiles);
    project.setProjectFilesList(list);

    Boolean ret = projectService.add(project);
    assertTrue(ret);

    List<ProjectEntity> projectList = projectService.queryPageList("ProjectServiceTest-UserName",
        "create_time", "desc", 0, 100);
    assertEquals(projectList.size(), 1);

    ProjectEntity projectDb = projectList.get(0);
    assertEquals(project.getDescription(), projectDb.getDescription());
    assertEquals(project.getName(), projectDb.getName());
    assertEquals(project.getType(), projectDb.getType());
    assertEquals(project.getUserName(), projectDb.getUserName());
    assertEquals(project.getVisibility(), projectDb.getVisibility());
    assertEquals(project.getCreateBy(), projectDb.getCreateBy());

    assertEquals(projectDb.getProjectFilesList().size(), 1);

    ProjectFilesEntity projectFilesDb = projectDb.getProjectFilesList().get(0);
    assertEquals(project.getId(), projectFilesDb.getProjectId());
    assertEquals(projectFiles.getFileContent(), projectFilesDb.getFileContent());
    assertEquals(projectFiles.getFileName(), projectFilesDb.getFileName());
    assertEquals(projectFiles.getCreateBy(), projectFilesDb.getCreateBy());
  }

  @Test
  public void updateByPrimaryKeySelective() throws Exception {
    ProjectFilesEntity projectFiles = new ProjectFilesEntity();
    projectFiles.setFileContent("ProjectServiceTest-FileContent");
    projectFiles.setFileName("ProjectServiceTest-FileName");
    projectFiles.setCreateBy("ProjectServiceTest-UserName");

    ProjectEntity project = new ProjectEntity();
    project.setDescription("ProjectServiceTest-Description");
    project.setName("ProjectServiceTest-ProjectName");
    project.setType("PROJECT_TYPE_NOTEBOOK");
    project.setUserName("ProjectServiceTest-UserName");
    project.setVisibility("PROJECT_VISIBILITY_PRIVATE");
    project.setCreateBy("ProjectServiceTest-UserName");
    List list = new ArrayList<ProjectFilesEntity>();
    list.add(projectFiles);
    project.setProjectFilesList(list);

    Boolean ret = projectService.add(project);
    assertTrue(ret);

    project.setName("update_projectName");
    project.setDescription("update_description");
    project.setVisibility("PROJECT_VISIBILITY_PUBLIC");
    project.setUpdateBy("project_updateBy");
    ProjectFilesEntity projectFilesUpdate = new ProjectFilesEntity();
    projectFilesUpdate.setFileContent("ProjectServiceTest-FileContent2");
    projectFilesUpdate.setFileName("ProjectServiceTest-FileName2");
    projectFilesUpdate.setCreateBy("ProjectServiceTest-UserName2");
    list.add(projectFilesUpdate);
    projectFiles.setFileName("update_fileName");
    projectFiles.setFileContent("update_fileContent");
    projectFiles.setUpdateBy("projectFiles_updateby");
    boolean editRet = projectService.updateByPrimaryKeySelective(project);
    assertTrue(editRet);
    List<ProjectEntity> projectList = projectService.queryPageList("ProjectServiceTest-UserName",
        "create_time", "desc", 0, 100);
    assertEquals(projectList.size(), 1);

    ProjectEntity projectDb = projectList.get(0);
    assertEquals(project.getName(), projectDb.getName());
    assertEquals(project.getDescription(), projectDb.getDescription());
    assertEquals(project.getVisibility(), projectDb.getVisibility());
    assertEquals(project.getUpdateBy(), projectDb.getUpdateBy());
    LOG.info("update_time:{}", projectDb.getUpdateTime());

    List<ProjectFilesEntity> projectFilesList = projectDb.getProjectFilesList();
    for (ProjectFilesEntity files : projectFilesList) {
      if (!files.getFileContent().equals("ProjectServiceTest-FileContent2")) {
        assertEquals(files.getFileName(), projectFiles.getFileName());
        assertEquals(files.getFileContent(), projectFiles.getFileContent());
        assertEquals(files.getUpdateBy(), projectFiles.getUpdateBy());
      }
    }
    assertEquals(projectFilesList.size(), 2);
  }

  @Test
  public void delete() throws Exception {
    ProjectFilesEntity projectFiles = new ProjectFilesEntity();
    projectFiles.setFileContent("ProjectServiceTest-FileContent");
    projectFiles.setFileName("ProjectServiceTest-FileName");
    projectFiles.setCreateBy("ProjectServiceTest-UserName");

    ProjectEntity project = new ProjectEntity();
    project.setDescription("ProjectServiceTest-Description");
    project.setName("ProjectServiceTest-ProjectName");
    project.setType("PROJECT_TYPE_NOTEBOOK");
    project.setUserName("ProjectServiceTest-UserName");
    project.setVisibility("PROJECT_VISIBILITY_PRIVATE");
    project.setCreateBy("ProjectServiceTest-UserName");
    List list = new ArrayList<ProjectFilesEntity>();
    list.add(projectFiles);
    project.setProjectFilesList(list);

    Boolean ret = projectService.add(project);
    assertTrue(ret);

    Boolean deleteRet = projectService.delete(project.getId());
    assertTrue(deleteRet);

    List<ProjectEntity> projectList = projectService.queryPageList("ProjectServiceTest-UserName",
        "create_time", "desc", 0, 100);
    assertEquals(projectList.size(), 0);

    ProjectFilesService projectFilesService = new ProjectFilesService();
    List<ProjectFilesEntity> projectFilesList = projectFilesService.queryList(project.getId());
    assertEquals(projectFilesList.size(), 0);
  }
}
