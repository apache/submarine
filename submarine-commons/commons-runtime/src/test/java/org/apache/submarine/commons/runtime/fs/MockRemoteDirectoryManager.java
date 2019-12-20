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

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Time;

import java.io.File;
import java.io.IOException;
import java.util.Objects;

public class MockRemoteDirectoryManager implements RemoteDirectoryManager {

  private static final String FAILED_TO_CREATE_DIRS_FORMAT_STRING =
      "Failed to create directories under path: %s";
  private static final String JOB_NAME_MUST_NOT_BE_NULL =
      "Job name must not be null!";
  private static final File STAGING_AREA = new File("target/_staging_area_");

  private File jobsParentDir;
  private File jobDir;
  private File modelParentDir;

  public MockRemoteDirectoryManager(String jobName) throws IOException {
    Objects.requireNonNull(jobName, JOB_NAME_MUST_NOT_BE_NULL);
    this.cleanup();
    this.jobsParentDir = initializeJobParentDir();
    this.jobDir = initializeJobDir(jobName);
    this.modelParentDir = initializeModelParentDir();
  }

  private void cleanup() throws IOException {
    FileUtils.deleteDirectory(STAGING_AREA);
  }

  private File initializeJobParentDir() throws IOException {
    File dir = new File(STAGING_AREA, String.valueOf(Time.monotonicNow()));
    if (!dir.mkdirs()) {
      throw new IOException(
          String.format(FAILED_TO_CREATE_DIRS_FORMAT_STRING,
              dir.getAbsolutePath()));
    }
    return dir;
  }

  private File initializeJobDir(String jobName) throws IOException {
    Objects.requireNonNull(jobsParentDir, "Job parent dir must not be null!");
    File dir = new File(jobsParentDir.getAbsolutePath(), jobName);

    if (!dir.exists() && !dir.mkdirs()) {
      throw new IOException(
          String.format(FAILED_TO_CREATE_DIRS_FORMAT_STRING,
              dir.getAbsolutePath()));
    }
    return dir;
  }

  private File initializeModelParentDir() throws IOException {
    File dir = new File(
        "target/_models_" + System.currentTimeMillis());
    if (!dir.exists() && !dir.mkdirs()) {
      throw new IOException(
          String.format(FAILED_TO_CREATE_DIRS_FORMAT_STRING,
              dir.getAbsolutePath()));
    }
    return dir;
  }

  @Override
  public Path getJobStagingArea(String jobName, boolean create)
      throws IOException {
    Objects.requireNonNull(jobName, JOB_NAME_MUST_NOT_BE_NULL);
    Objects.requireNonNull(jobDir, JOB_NAME_MUST_NOT_BE_NULL);
    this.jobDir = initializeJobDir(jobName);
    if (create && !jobDir.exists()) {
      if (!jobDir.mkdirs()) {
        throw new IOException(
            String.format(FAILED_TO_CREATE_DIRS_FORMAT_STRING,
                jobDir.getAbsolutePath()));
      }
    }
    return new Path(jobDir.getAbsolutePath());
  }

  @Override
  public Path getJobCheckpointDir(String jobName, boolean create)
      throws IOException {
    return new Path("s3://generated_checkpoint_dir");
  }

  @Override
  public Path getModelDir(String modelName, boolean create)
      throws IOException {
    File modelDir = new File(modelParentDir.getAbsolutePath(), modelName);
    if (create) {
      if (!modelDir.exists() && !modelDir.mkdirs()) {
        throw new IOException("Failed to mkdirs for "
            + modelDir.getAbsolutePath());
      }
    }
    return new Path(modelDir.getAbsolutePath());
  }

  @Override
  public FileSystem getDefaultFileSystem() throws IOException {
    return FileSystem.getLocal(new Configuration());
  }

  @Override
  public FileSystem getFileSystemByUri(String uri) throws IOException {
    return getDefaultFileSystem();
  }

  @Override
  public Path getUserRootFolder() throws IOException {
    return new Path("s3://generated_root_dir");
  }

  @Override
  public boolean isDir(String uri) throws IOException {
    return getDefaultFileSystem().getFileStatus(
        new Path(convertToStagingPath(uri))).isDirectory();
  }

  @Override
  public boolean isRemote(String uri) throws IOException {
    String scheme = new Path(uri).toUri().getScheme();
    if (null == scheme) {
      return false;
    }
    return !scheme.startsWith("file://");
  }

  private String convertToStagingPath(String uri) throws IOException {
    if (isRemote(uri)) {
      String dirName = new Path(uri).getName();
      return this.jobDir.getAbsolutePath()
          + "/" + dirName;
    }
    return uri;
  }

  /**
   * We use staging dir as mock HDFS dir.
   * */
  @Override
  public boolean copyRemoteToLocal(String remoteUri, String localUri)
      throws IOException {
    // mock the copy from HDFS into a local copy
    Path remoteToLocalDir = new Path(convertToStagingPath(remoteUri));
    File old = new File(convertToStagingPath(localUri));
    if (old.isDirectory() && old.exists()) {
      if (!FileUtil.fullyDelete(old)) {
        throw new IOException("Cannot delete temp dir:"
            + old.getAbsolutePath());
      }
    }
    return FileUtil.copy(getDefaultFileSystem(), remoteToLocalDir,
        new File(localUri), false,
        getDefaultFileSystem().getConf());
  }

  @Override
  public boolean existsRemoteFile(Path uri) throws IOException {
    String fakeLocalFilePath = this.jobDir.getAbsolutePath()
        + "/" + uri.getName();
    return new File(fakeLocalFilePath).exists();
  }

  @Override
  public FileStatus getRemoteFileStatus(Path p) throws IOException {
    return getDefaultFileSystem().getFileStatus(new Path(
        convertToStagingPath(p.toUri().toString())));
  }

  @Override
  public long getRemoteFileSize(String uri) throws IOException {
    // 5 byte for this file to test
    if (uri.equals("https://a/b/1.patch")) {
      return 5;
    }
    return 100 * 1024 * 1024;
  }

  public void setJobDir(File jobDir) {
    this.jobDir = jobDir;
  }
}
