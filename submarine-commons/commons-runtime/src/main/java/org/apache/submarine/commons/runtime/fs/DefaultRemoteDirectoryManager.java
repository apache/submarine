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

package org.apache.submarine.commons.runtime.fs;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.submarine.commons.runtime.ClientContext;

import java.io.File;
import java.io.IOException;
import java.net.URI;

/**
 * Manages remote directories for staging, log, etc.
 * TODO(keqiu), need to properly handle permission / name validation, etc.
 */
public class DefaultRemoteDirectoryManager implements RemoteDirectoryManager {
  private FileSystem fs;
  private Configuration conf;

  public DefaultRemoteDirectoryManager(ClientContext context) {
    this.conf = context.getYarnConfig();
    try {
      this.fs = FileSystem.get(context.getYarnConfig());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Path getJobStagingArea(String jobName, boolean create)
      throws IOException {
    Path staging = new Path(getJobRootFolder(jobName), "staging");
    if (create) {
      createFolderIfNotExist(staging);
    }

    // Get a file status to make sure it is a absolute path.
    FileStatus fStatus = fs.getFileStatus(staging);
    return fStatus.getPath();
  }

  @Override
  public Path getJobCheckpointDir(String jobName, boolean create)
      throws IOException {
    Path jobDir = new Path(getJobStagingArea(jobName, create),
        "checkpoint_path");
    if (create) {
      createFolderIfNotExist(jobDir);
    }
    return jobDir;
  }

  @Override
  public Path getModelDir(String modelName, boolean create)
      throws IOException {
    Path modelDir = new Path(new Path("submarine", "models"), modelName);
    if (create) {
      createFolderIfNotExist(modelDir);
    }
    return modelDir;
  }

  @Override
  public FileSystem getDefaultFileSystem() {
    return fs;
  }

  @Override
  public FileSystem getFileSystemByUri(String uri) throws IOException {
    return FileSystem.get(URI.create(uri), conf);
  }

  @Override
  public Path getUserRootFolder() throws IOException {
    Path rootPath = new Path("submarine", "jobs");
    createFolderIfNotExist(rootPath);
    // Get a file status to make sure it is a absolute path.
    FileStatus fStatus = fs.getFileStatus(rootPath);
    return fStatus.getPath();
  }

  @Override
  public boolean isDir(String uri) throws IOException {
    if (isRemote(uri)) {
      return getFileSystemByUri(uri).getFileStatus(new Path(uri)).isDirectory();
    }
    return new File(uri).isDirectory();
  }

  @Override
  public boolean isRemote(String uri) {
    String scheme = new Path(uri).toUri().getScheme();
    if (null == scheme) {
      return false;
    }
    return !scheme.startsWith("file://");
  }

  @Override
  public boolean copyRemoteToLocal(String remoteUri, String localUri)
      throws IOException {
    // Delete old to avoid failure in FileUtil.copy
    File old = new File(localUri);
    if (old.exists()) {
      if (!FileUtil.fullyDelete(old)) {
        throw new IOException("Failed to delete dir:"
            + old.getAbsolutePath());
      }
    }
    return FileUtil.copy(getFileSystemByUri(remoteUri), new Path(remoteUri),
        new File(localUri), false,
        conf);
  }

  @Override
  public boolean existsRemoteFile(String remoteUri) throws IOException {
    return getFileSystemByUri(remoteUri).exists(new Path(remoteUri));
  }

  @Override
  public FileStatus getRemoteFileStatus(String remoteUri) throws IOException {
    return getFileSystemByUri(remoteUri).getFileStatus(new Path(remoteUri));
  }

  @Override
  public long getRemoteFileSize(String uri) throws IOException {
    return getFileSystemByUri(uri)
        .getContentSummary(new Path(uri)).getSpaceConsumed();
  }

  private Path getJobRootFolder(String jobName) throws IOException {
    Path userRoot = getUserRootFolder();
    Path jobRootPath = new Path(userRoot, jobName);
    createFolderIfNotExist(jobRootPath);
    // Get a file status to make sure it is a absolute path.
    FileStatus fStatus = fs.getFileStatus(jobRootPath);
    return fStatus.getPath();
  }

  private void createFolderIfNotExist(Path path) throws IOException {
    if (!fs.exists(path)) {
      if (!fs.mkdirs(path)) {
        throw new IOException("Failed to create folder=" + path);
      }
    }
  }
}
