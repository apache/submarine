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
package org.apache.submarine.server.utils;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.api.PullResult;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.dircache.DirCache;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.transport.CredentialsProvider;
import org.eclipse.jgit.transport.PushResult;
import org.eclipse.jgit.transport.URIish;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

public class GitUtils {
  private static final Logger LOG = LoggerFactory.getLogger(GitUtils.class);

  /**
   * To execute clone command.
   * @param remotePath the URI of the remote repository
   * @param localPath the path of the local repository
   * @param token the GitHub access token
   * @param branch the git branch to use
   */
  public void clone(String remotePath, String localPath, String token, String branch) {
    // Clone the code base command
    // Sets the token on the remote server
    CredentialsProvider credentialsProvider =
        new UsernamePasswordCredentialsProvider("PRIVATE-TOKEN", token);

    try (Git git = Git.cloneRepository().setURI(remotePath) // Set remote URI
            .setBranch(branch) // Set the branch down from clone
            .setDirectory(new File(localPath)) // Set the download path
            .setCredentialsProvider(credentialsProvider) // Set permission validation
            .call();) {

      LOG.info("git.tag(): {}", git.tag());
    } catch (GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  /**
   * To execute add command.
   * @param localPath the path of the local repository
   * @param fileName the file name you want to add to the git stage area (relative path)
   * @return DirCache
   */
  public DirCache add(String localPath, String fileName) {
    // Git repository address

    // File myfile = new File(localPath + fileName);
    File myfile = new File(localPath, fileName);

    if (!myfile.exists()) {
      myfile.getParentFile().mkdirs();
      try {
        myfile.createNewFile();
      } catch (IOException e) {
        LOG.error(e.getMessage(), e);
      }
    }

    DirCache dirCache = null;
    try (Git git = Git.open(new File(localPath))) {
      // Add files
      dirCache = git.add().addFilepattern(fileName.substring(1)).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return dirCache;
  }

  /**
   * To execute rm command.
   * @param localPath the path of the local repository
   * @param fileName the file name you want to remove from the git repo (relative path)
   * @return DirCache
   */
  public DirCache rm(String localPath, String fileName) {
    DirCache dirCache = null;
    // Git repository address
    try (Git git = Git.open(new File(localPath))) {
      // rm files
      dirCache = git.rm().addFilepattern(fileName.substring(1)).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return dirCache;
  }


  /**
   * To execute reset command.
   * @param localPath the path of the local repository
   * @param fileName the file name you want to reset (relative path)
   */
  public void reset(String localPath, String fileName) {
    // Git repository address
    try (Git git = Git.open(new File(localPath))) {
      // reset files
      git.reset().addPath(fileName.substring(1)).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  /**
   * To execute pull command.
   * @param localPath the path of the local repository
   * @param token the GitHub access token
   * @return PullResult
   */
  public PullResult pull(String localPath, String token, String branch) {
    CredentialsProvider credentialsProvider =
        new UsernamePasswordCredentialsProvider("PRIVATE-TOKEN", token);
    PullResult pullResult = null;
    // Git repository address
    try (Git git = Git.open(new File(localPath))) {
      pullResult = git.pull().setRemoteBranchName(branch).
          setCredentialsProvider(credentialsProvider).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return pullResult;
  }

  /**
   * To execute commit command.
   * @param localPath the path of the local repository
   * @return RevCommit
   */
  public RevCommit commit(String localPath, String message) {
    RevCommit revCommit = null;
    // Git repository address
    try (Git git = Git.open(new File(localPath))) {
      // Submit code
      revCommit = git.commit().setMessage(message).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return revCommit;
  }

  /**
   * To execute push command.
   * @param localPath the path of the local repository
   * @param token the GitHub access token
   * @param remote the URI of remote repository
   * @return Iterable<PushResult>
   */
  public Iterable<PushResult> push(String localPath, String token, String remote) {
    CredentialsProvider credentialsProvider =
        new UsernamePasswordCredentialsProvider("PRIVATE-TOKEN", token);
    Iterable<PushResult> iterable = null;
    // Git repository address
    try (Git git = Git.open(new File(localPath))) {
      iterable = git.push().setRemote(remote).setCredentialsProvider(credentialsProvider).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return iterable;
  }

  /**
   * To execute branchCreate command.
   * @param localPath the path of the local repository
   * @param branchName the branch you want to create
   * @return Ref
   */
  public Ref branchCreate(String localPath, String branchName) {
    Ref result = null;
    try (Git git = Git.open(new File(localPath))) {
      ListBranchCommand listBranchCommand = git.branchList();
      List<Ref> list = listBranchCommand.call();
      boolean existsBranch = false;

      for (Ref ref : list) {
        if (ref.getName().endsWith(branchName)) {
          existsBranch = true;
          break;
        }
      }

      if (!existsBranch) {
        // Create branch
        result = git.branchCreate().setName(branchName).call();
      } else {
        LOG.warn("{} already exists.", branchName);
      }
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return result;
  }

  /**
   * To execute branchDelete command.
   * @param localPath the path of the local repository
   * @param branchName the branch you want to delete
   * @return List<String>
   */
  public List<String> branchDelete(String localPath, String branchName) {
    List<String> list = null;
    try (Git git = Git.open(new File(localPath))) {
      list = git.branchDelete().setForce(true).setBranchNames(branchName).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return list;
  }

  /**
   * To execute checkout command.
   * @param localPath the path of the local repository
   * @param branchName the branch you want to checkout to
   * @return Ref
   */
  public Ref checkout(String localPath, String branchName) {
    Ref result = null;
    try (Git git = Git.open(new File(localPath))) {
      result = git.checkout().setName(branchName).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }

    return result;
  }

  /**
   * To execute rebase command.
   * @param localPath the path of the local repository
   * @param branchName the branch you want to rebase from
   * @param upstreamName the name of upstream repository
   */
  public void rebase(String localPath, String branchName, String upstreamName) {
    try (Git git = Git.open(new File(localPath))) {
      git.rebase().setUpstream(branchName).setUpstreamName(upstreamName).call();
    } catch (IOException | GitAPIException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  /**
   * To execute remoteAdd command.
   * @param localPath the path of the local repository
   * @param uri the URI prefix of the remote repository
   * @param remoteName the name of the remote repository
   */
  public void remoteAdd(String localPath, String uri, String remoteName) {
    try (Git git = Git.open(new File(localPath))) {
      URIish urIish = new URIish(uri);
      git.remoteAdd().setName(remoteName).setUri(urIish).call();
    } catch (IOException | GitAPIException | URISyntaxException e) {
      LOG.error(e.getMessage(), e);
    }
  }
}
