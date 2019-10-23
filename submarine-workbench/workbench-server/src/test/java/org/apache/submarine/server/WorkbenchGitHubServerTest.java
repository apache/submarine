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
package org.apache.submarine.server;

import com.google.common.collect.Lists;
import com.google.gson.Gson;
import org.eclipse.jgit.api.PullResult;
import org.eclipse.jgit.dircache.DirCache;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.transport.PushResult;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static org.apache.submarine.database.utils.HttpRequestUtil.sendHttpRequest;

public class WorkbenchGitHubServerTest {

  private static final Logger LOG = LoggerFactory.getLogger(WorkbenchGitHubServerTest.class);
  private static final String TOKEN = "30f6bf0d557f22e69106efa9aa6c2b37ea6ac7cc";
  private static final String OWNER = "submarine-thirdparty";
  private static final String REPO = "submarine_git_test";
  WorkbenchGitHubServer workbenchGitHubServer = new WorkbenchGitHubServer();
  private static final String REMOTE_PATH = "https://github.com/" + OWNER + "/" + REPO + ".git";
  private static final String BRANCHNAME = "Branch-1";
  private static final String LOCALPATH =
      WorkbenchGitHubServerTest.class.getResource("").toString().substring(6) + REPO;

  @After
  public void restoreAllOperations() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      Map<String, String> map = new HashMap<>();
      map.put("Authorization", "token " + token);
      LOG.info("gitToken: {}", token);
      sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO + "/subscription",
          map, null, "DELETE");
      sendHttpRequest("https://api.github.com/user/starred/" + OWNER + "/" + REPO + "",
          map, null, "DELETE");

      String result = sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO
          + "/branches", map, null, "GET");
      Gson gson = new Gson();
      List<Map<String, String>> list = gson.fromJson(result, List.class);
      for (Map<String, String> stringStringMap : list) {
        if (BRANCHNAME.equals(stringStringMap.get("name"))) {
          sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO
              + "/git/refs/heads/" + BRANCHNAME, map, null, "DELETE");
        }
      }
      LOG.info("LOCALPATH: {}", LOCALPATH);
      deleteDirectory(LOCALPATH);
    }
  }

  public void deleteDirectory(String dir) {
    File dirFile = new File(dir);
    File[] files = dirFile.listFiles();
    if (files != null) {
      for (File file : files) {
        if (file.isFile()) {
          new File(file.getAbsolutePath()).delete();
        } else if (file.isDirectory()) {
          deleteDirectory(file.getAbsolutePath());
        }
      }
      dirFile.delete();
    }
  }

  @Test
  public void addWatching() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      Map<String, String> map = new HashMap<>();
      map.put("Authorization", "token " + token);
      LOG.info("gitToken: {}", token);
      sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO + "/subscription",
          map, null, "PUT");

      String result = sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO
          + "/subscribers", null, null, "GET");
      LOG.info("Watching result: {}", result);

      Gson gson = new Gson();
      List<Map<String, String>> list = gson.fromJson(result, List.class);
      assertEquals(1, list.size());
      String login = list.get(0).get("login");
      LOG.info("login: {}", login);
      assertEquals(OWNER, login);
    }
  }

  @Test
  public void deleteWatching() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      Map<String, String> map = new HashMap<>();
      map.put("Authorization", "token " + token);
      LOG.info("gitToken: {}", token);
      sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO + "/subscription",
          map, null, "PUT");

      sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO + "/subscription",
          map, null, "DELETE");

      String result = sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO
          + "/subscribers", null, null, "GET");
      LOG.info("result: {}", result);

      Gson gson = new Gson();
      List<Map<String, String>> list = gson.fromJson(result, List.class);
      assertEquals(0, list.size());
    }
  }

  @Test
  public void addStarring() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      Map<String, String> map = new HashMap<>();
      map.put("Authorization", "token " + token);
      LOG.info("gitToken: {}", token);
      sendHttpRequest("https://api.github.com/user/starred/" + OWNER + "/" + REPO + "",
          map, null, "PUT");

      String result = sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO
          + "/stargazers", null, null, "GET");
      LOG.info("Starring result: {}", result);

      Gson gson = new Gson();
      List<Map<String, String>> list = gson.fromJson(result, List.class);
      assertEquals(1, list.size());
      String login = list.get(0).get("login");
      LOG.info("login: {}", login);
      assertEquals(OWNER, login);
    }
  }

  @Test
  public void deleteStarring() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      Map<String, String> map = new HashMap<>();
      map.put("Authorization", "token " + token);
      LOG.info("gitToken: {}", token);
      sendHttpRequest("https://api.github.com/user/starred/" + OWNER + "/" + REPO + "",
          map, null, "PUT");

      sendHttpRequest("https://api.github.com/user/starred/" + OWNER + "/" + REPO + "",
          map, null, "DELETE");

      String result = sendHttpRequest("https://api.github.com/repos/" + OWNER + "/" + REPO
          + "/stargazers", null, null, "GET");
      LOG.info("result: {}", result);

      Gson gson = new Gson();
      List<Map<String, String>> list = gson.fromJson(result, List.class);
      assertEquals(0, list.size());
    }
  }

  @Before
  public void cloneTest() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      workbenchGitHubServer.clone(REMOTE_PATH, LOCALPATH, token, "master");
      File dirFile = new File(LOCALPATH);
      File[] files = dirFile.listFiles();
      assertEquals(1, files.length);
      assertEquals(".git", files[0].getName());
    }
  }

  @Test
  public void addAndRest() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      DirCache dirCache = workbenchGitHubServer.add(LOCALPATH, "/aa/bb/log4j.properties");
      assertEquals(1, dirCache.getEntryCount());

      workbenchGitHubServer.reset(LOCALPATH, "/aa/bb/log4j.properties");
    }
  }

  @Test
  public void pull() {
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      workbenchGitHubServer.add(LOCALPATH, "/log4j.properties");
      workbenchGitHubServer.commit(LOCALPATH, "add new file.");
      Iterable<PushResult> iterable = workbenchGitHubServer.push(LOCALPATH, token, REMOTE_PATH);
      assertEquals(1, Lists.newArrayList(iterable).size());

      PullResult pullResult = workbenchGitHubServer.pull(LOCALPATH, token, "master");
      assertEquals(1, pullResult.getFetchResult().getTrackingRefUpdates().size());

      workbenchGitHubServer.rm(LOCALPATH, "/log4j.properties");
      workbenchGitHubServer.commit(LOCALPATH, "add new file.");
      workbenchGitHubServer.push(LOCALPATH, token, REMOTE_PATH);
    }
  }

  @Test
  public void branchCreateAndCheckout() {
    workbenchGitHubServer.commit(LOCALPATH, "add new file.");
    String token = System.getenv("gitToken");
    token = TOKEN;
    if (token == null) {
      LOG.warn("Token not set!");
      return;
    } else {
      Ref ref = workbenchGitHubServer.branchCreate(LOCALPATH, BRANCHNAME);
      assertEquals(true, ref.getName().endsWith(BRANCHNAME));

      ref = workbenchGitHubServer.checkout(LOCALPATH, BRANCHNAME);
      assertEquals(true, ref.getName().endsWith(BRANCHNAME));
    }
  }
}
