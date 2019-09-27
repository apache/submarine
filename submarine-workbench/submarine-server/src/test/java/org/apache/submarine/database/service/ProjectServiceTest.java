/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.database.service;

import org.apache.submarine.database.entity.Project;
import org.apache.submarine.database.entity.ProjectFiles;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class ProjectServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(TeamServiceTest.class);
  ProjectService projectService = new ProjectService();

  @After
  public void removeAllRecord() throws Exception {
    List<Project> projectList = projectService.queryPageList(null, "create_time", "desc", 0, 100);
    LOG.info("projectList.size():{}", projectList.size());
    for (Project project : projectList) {
      projectService.delete(project.getId());
    }
  }

  @Test
  public void queryPageList() throws Exception {
    ProjectFiles projectFiles = new ProjectFiles();
    projectFiles.setFileContent("ProjectServiceTest-FileContent");
    projectFiles.setFileName("ProjectServiceTest-FileName");
    projectFiles.setCreateBy("ProjectServiceTest-UserName");

    Project project = new Project();
    project.setDescription("ProjectServiceTest-Description");
    project.setProjectName("ProjectServiceTest-ProjectName");
    project.setType(1);
    project.setUserName("ProjectServiceTest-UserName");
    project.setVisibility(1);
    project.setCreateBy("ProjectServiceTest-UserName");
    List list = new ArrayList<ProjectFiles>();
    list.add(projectFiles);
    project.setProjectFilesList(list);

    Boolean ret = projectService.add(project);
    assertTrue(ret);

    List<Project> projectList = projectService.queryPageList("ProjectServiceTest-UserName",
        "create_time", "desc", 0, 100);
    assertEquals(projectList.size(), 1);

    Project project_db = projectList.get(0);
    assertEquals(project.getDescription(), project_db.getDescription());
    assertEquals(project.getProjectName(), project_db.getProjectName());
    assertEquals(project.getType(), project_db.getType());
    assertEquals(project.getUserName(), project_db.getUserName());
    assertEquals(project.getVisibility(), project_db.getVisibility());
    assertEquals(project.getCreateBy(), project_db.getCreateBy());

    assertEquals(project_db.getProjectFilesList().size(), 1);

    ProjectFiles projectFiles_db = project_db.getProjectFilesList().get(0);
    assertEquals(project.getId(), projectFiles_db.getProjectId());
    assertEquals(projectFiles.getFileContent(), projectFiles_db.getFileContent());
    assertEquals(projectFiles.getFileName(), projectFiles_db.getFileName());
    assertEquals(projectFiles.getCreateBy(), projectFiles_db.getCreateBy());
  }

  @Test
  public void updateByPrimaryKeySelective() throws Exception {
    ProjectFiles projectFiles = new ProjectFiles();
    projectFiles.setFileContent("ProjectServiceTest-FileContent");
    projectFiles.setFileName("ProjectServiceTest-FileName");
    projectFiles.setCreateBy("ProjectServiceTest-UserName");

    Project project = new Project();
    project.setDescription("ProjectServiceTest-Description");
    project.setProjectName("ProjectServiceTest-ProjectName");
    project.setType(1);
    project.setUserName("ProjectServiceTest-UserName");
    project.setVisibility(1);
    project.setCreateBy("ProjectServiceTest-UserName");
    List list = new ArrayList<ProjectFiles>();
    list.add(projectFiles);
    project.setProjectFilesList(list);

    Boolean ret = projectService.add(project);
    assertTrue(ret);

    project.setProjectName("update_projectName");
    project.setDescription("update_description");
    project.setVisibility(2);
    project.setUpdateBy("project_updateBy");
    ProjectFiles projectFiles_update = new ProjectFiles();
    projectFiles_update.setFileContent("ProjectServiceTest-FileContent2");
    projectFiles_update.setFileName("ProjectServiceTest-FileName2");
    projectFiles_update.setCreateBy("ProjectServiceTest-UserName2");
    list.add(projectFiles_update);
    projectFiles.setFileName("update_fileName");
    projectFiles.setFileContent("update_fileContent");
    projectFiles.setUpdateBy("projectFiles_updateby");
    boolean editRet = projectService.updateByPrimaryKeySelective(project);
    assertTrue(editRet);
    List<Project> projectList = projectService.queryPageList("ProjectServiceTest-UserName",
        "create_time", "desc", 0, 100);
    assertEquals(projectList.size(), 1);

    Project project_db = projectList.get(0);
    assertEquals(project.getProjectName(), project_db.getProjectName());
    assertEquals(project.getDescription(), project_db.getDescription());
    assertEquals(project.getVisibility(), project_db.getVisibility());
    assertEquals(project.getUpdateBy(), project_db.getUpdateBy());
    LOG.info("update_time:{}", project_db.getUpdateTime());

    List<ProjectFiles> projectFilesList = project_db.getProjectFilesList();
    for (ProjectFiles files : projectFilesList) {
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
    ProjectFiles projectFiles = new ProjectFiles();
    projectFiles.setFileContent("ProjectServiceTest-FileContent");
    projectFiles.setFileName("ProjectServiceTest-FileName");
    projectFiles.setCreateBy("ProjectServiceTest-UserName");

    Project project = new Project();
    project.setDescription("ProjectServiceTest-Description");
    project.setProjectName("ProjectServiceTest-ProjectName");
    project.setType(1);
    project.setUserName("ProjectServiceTest-UserName");
    project.setVisibility(1);
    project.setCreateBy("ProjectServiceTest-UserName");
    List list = new ArrayList<ProjectFiles>();
    list.add(projectFiles);
    project.setProjectFilesList(list);

    Boolean ret = projectService.add(project);
    assertTrue(ret);

    Boolean deleteRet = projectService.delete(project.getId());
    assertTrue(deleteRet);

    List<Project> projectList = projectService.queryPageList("ProjectServiceTest-UserName",
        "create_time", "desc", 0, 100);
    assertEquals(projectList.size(), 0);

    ProjectFilesService projectFilesService = new ProjectFilesService();
    List<ProjectFiles> projectFilesList = projectFilesService.queryList(project.getId());
    assertEquals(projectFilesList.size(), 0);
  }
}
