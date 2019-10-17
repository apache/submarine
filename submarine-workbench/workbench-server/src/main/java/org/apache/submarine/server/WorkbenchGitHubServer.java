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

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.internal.storage.file.FileRepository;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.transport.CredentialsProvider;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class WorkbenchGitHubServer {
  private static final Logger LOG = LoggerFactory.getLogger(WorkbenchGitHubServer.class);

  /**
   * To execute clone command
   * @param remotePath
   * @param localPath
   * @param token
   * @param branch
   * @throws GitAPIException
   */
  public void clone(String remotePath, String localPath, String token, String branch)
      throws GitAPIException {
    // Clone the code base command
    // Sets the token on the remote server
    CredentialsProvider credentialsProvider =
        new UsernamePasswordCredentialsProvider("PRIVATE-TOKEN", token);

    Git git = Git.cloneRepository().setURI(remotePath) // Set remote URI
        .setBranch(branch) // Set the branch down from clone
        .setDirectory(new File(localPath)) // Set the download path
        .setCredentialsProvider(credentialsProvider) // Set permission validation
        .call();
    LOG.info("git.tag(): {}", git.tag());
  }

  /**
   * To execute add command
   * @param localPath
   * @param fileName
   * @throws IOException
   * @throws GitAPIException
   */
  public void add(String localPath, String fileName) throws IOException, GitAPIException {
    File myfile = new File(localPath + fileName);
    myfile.createNewFile();
    // Git repository address
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
      // Add files
      git.add().addFilepattern(fileName).call();
    }
  }

  /**
   * To execute pull command
   * @param localPath
   * @param token
   * @throws IOException
   * @throws GitAPIException
   */
  public void pull(String localPath, String token, String branch) throws IOException, GitAPIException {
    CredentialsProvider credentialsProvider =
        new UsernamePasswordCredentialsProvider("PRIVATE-TOKEN", token);
    // Git repository address
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
      git.pull().setRemoteBranchName(branch).
          setCredentialsProvider(credentialsProvider).call();
    }
  }

  /**
   * To execute commit command
   * @param localPath
   * @throws IOException
   * @throws GitAPIException
   */
  public void commit(String localPath, String message) throws IOException, GitAPIException {
    // Git repository address
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
      // Submit code
      git.commit().setMessage(message).call();
    }
  }

  /**
   * To execute push command
   * @param localPath
   * @param token
   * @throws IOException
   * @throws GitAPIException
   */
  public void push(String localPath, String token, String remote) throws IOException, GitAPIException {
    CredentialsProvider credentialsProvider =
        new UsernamePasswordCredentialsProvider("PRIVATE-TOKEN", token);
    // Git repository address
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
      git.push().setRemote(remote).setCredentialsProvider(credentialsProvider).call();
    }
  }

  /**
   * To execute branchCreate command
   * @param localPath
   * @param branchName
   * @throws IOException
   * @throws GitAPIException
   */
  public void branchCreate(String localPath, String branchName) throws IOException, GitAPIException {
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
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
        git.branchCreate().setName(branchName).call();
      } else {
        LOG.warn("{} already exists.", branchName);
      }
    }
  }

  /**
   * To execute checkout command
   * @param localPath
   * @param branchName
   * @throws IOException
   * @throws GitAPIException
   */
  public void checkout(String localPath, String branchName) throws IOException, GitAPIException {
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
      git.checkout().setName(branchName).call();
    }
  }

  /**
   * To execute rebase command
   * @param localPath
   * @param branchName
   * @param upstreamName
   * @throws IOException
   * @throws GitAPIException
   */
  public void rebase(String localPath, String branchName, String upstreamName)
      throws IOException, GitAPIException {
    try (Git git = new Git(new FileRepository(localPath + ".git"))) {
      git.rebase().setUpstream(branchName).setUpstreamName(upstreamName).call();
    }
  }
}
