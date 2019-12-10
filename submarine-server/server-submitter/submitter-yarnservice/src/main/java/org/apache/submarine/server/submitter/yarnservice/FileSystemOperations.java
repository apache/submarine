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

package org.apache.submarine.server.submitter.yarnservice;

import com.google.common.annotations.VisibleForTesting;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.yarn.service.api.records.Component;
import org.apache.hadoop.yarn.service.api.records.ConfigFile;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.runtime.conf.SubmarineLogs;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.submarine.server.submitter.yarnservice.utils.ZipUtilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Contains methods to perform file system operations. Almost all of the methods
 * are regular non-static methods as the operations are performed with the help
 * of a {@link RemoteDirectoryManager} instance passed in as a constructor
 * dependency. Please note that some operations require to read config settings
 * as well, so that we have Submarine and YARN config objects as dependencies as
 * well.
 */
public class FileSystemOperations {
  private static final String TEMP_DIR = System.getProperty("java.io.tmpdir");
  private static final long BYTES_IN_MB = 1024 * 1024;

  private static class DownloadResult {
    private final String srcPath;
    private final String dstPath;
    private final String suffix;
    private boolean remote;

    DownloadResult(String srcPath, String dstPath, String suffix,
        boolean remote) {
      this.srcPath = srcPath;
      this.dstPath = dstPath;
      this.suffix = suffix;
      this.remote = remote;
    }
  }

  private static final Logger LOG =
      LoggerFactory.getLogger(FileSystemOperations.class);
  private final SubmarineConfiguration submarineConfig;
  private final Configuration yarnConfig;

  private Set<Path> uploadedFiles = new HashSet<>();
  private RemoteDirectoryManager remoteDirectoryManager;

  public FileSystemOperations(ClientContext clientContext) {
    this.remoteDirectoryManager = clientContext.getRemoteDirectoryManager();
    this.submarineConfig = clientContext.getSubmarineConfig();
    this.yarnConfig = clientContext.getYarnConfig();
  }

  /**
   * May download a remote URI (file or directory) and zip it if asked.
   * Skips downloading local directories.
   * Remote URI can be a local dir or remote HDFS dir,
   * S3 file or directory, etc.
   */
  public String downloadAndZip(String remoteDir, String destFileName)
      throws IOException {
    DownloadResult downloadResult = downloadInternal(remoteDir, destFileName);
    String zipFileUri = zipLocalDirectory(downloadResult);

    if (downloadResult.remote) {
      deleteFiles(downloadResult.srcPath);
    }
    return zipFileUri;
  }

  public String download(String remoteDir, String zipFileName)
      throws IOException {
    DownloadResult downloadResult = downloadInternal(remoteDir, zipFileName);
    return downloadResult.srcPath;
  }

  private DownloadResult downloadInternal(String remoteDir, String destFileName)
      throws IOException {
    String suffix;
    String srcDir = remoteDir;

    String destFilePath = getFilePathInTempDir(destFileName);

    boolean remote = remoteDirectoryManager.isRemote(remoteDir);
    if (remote) {
      // Append original modification time and size to zip file name
      FileStatus status =
          remoteDirectoryManager.getRemoteFileStatus(new Path(remoteDir));
      suffix = getSuffixOfRemoteDirectory(remoteDir, status);
      // Download them to temp dir
      downloadRemoteFile(remoteDir, destFilePath);
      srcDir = destFilePath;
    } else {
      File localDir = new File(remoteDir);
      suffix = getSuffixOfLocalDirectory(localDir);
    }
    return new DownloadResult(srcDir, destFilePath, suffix, remote);
  }

  private String getFilePathInTempDir(String zipFileName) {
    return new File(TEMP_DIR, zipFileName).getAbsolutePath();
  }

  private String zipLocalDirectory(DownloadResult downloadResult)
      throws IOException {
    String dstFileName = downloadResult.dstPath +
        downloadResult.suffix + ".zip";
    return ZipUtilities.zipDir(downloadResult.srcPath,
        dstFileName);
  }

  private String getSuffixOfRemoteDirectory(String remoteDir,
      FileStatus status) throws IOException {
    return getSuffixOfDirectory(status.getModificationTime(),
        remoteDirectoryManager.getRemoteFileSize(remoteDir));
  }

  private String getSuffixOfLocalDirectory(File localDir) {
    return getSuffixOfDirectory(localDir.lastModified(), localDir.length());
  }

  private String getSuffixOfDirectory(long modificationTime, long size) {
    return "_" + modificationTime + "-" + size;
  }

  private void downloadRemoteFile(String remoteDir, String zipDirPath)
      throws IOException {
    boolean downloaded =
        remoteDirectoryManager.copyRemoteToLocal(remoteDir, zipDirPath);
    if (!downloaded) {
      throw new IOException("Failed to download Internal files from "
          + remoteDir);
    }
    LOG.info("Downloaded remote file: {} to this local path: {}",
        remoteDir, zipDirPath);
  }


  public void deleteFiles(String localUri) {
    boolean success = FileUtil.fullyDelete(new File(localUri));
    if (!success) {
      LOG.warn("Failed to delete {}", localUri);
    }
    LOG.info("Deleted {}", localUri);
  }

  @VisibleForTesting
  public void uploadToRemoteFileAndLocalizeToContainerWorkDir(Path stagingDir,
      String fileToUpload, String destFilename, Component comp)
      throws IOException {
    Path uploadedFilePath = uploadToRemoteFile(stagingDir, fileToUpload);
    locateRemoteFileToContainerWorkDir(destFilename, comp, uploadedFilePath);
  }

  private void locateRemoteFileToContainerWorkDir(String destFilename,
      Component comp, Path uploadedFilePath)
      throws IOException {
    FileSystem fs = FileSystem.get(yarnConfig);

    FileStatus fileStatus = fs.getFileStatus(uploadedFilePath);
    LOG.info("Uploaded file path: " + fileStatus.getPath());
    ConfigFile configFile = new ConfigFile()
        .srcFile(fileStatus.getPath().toUri().toString())
        .destFile(destFilename)
        .type(ConfigFile.TypeEnum.STATIC);
    addFilesToComponent(comp, configFile);
  }

  private void addFilesToComponent(Component comp, ConfigFile... configFiles) {
    for (ConfigFile configFile : configFiles) {
      comp.getConfiguration().getFiles().add(configFile);
    }
  }

  public Path uploadToRemoteFile(Path stagingDir, String fileToUpload) throws
      IOException {
    FileSystem fs = remoteDirectoryManager.getDefaultFileSystem();

    // Upload to remote FS under staging area
    File localFile = new File(fileToUpload);
    if (!localFile.exists()) {
      throw new FileNotFoundException(
          "Trying to upload file " + localFile.getAbsolutePath()
              + " to remote, but could not find local file!");
    }
    String filename = localFile.getName();

    Path uploadedFilePath = new Path(stagingDir, filename);
    if (!uploadedFiles.contains(uploadedFilePath)) {
      if (SubmarineLogs.isVerbose()) {
        LOG.info("Copying local file " + fileToUpload + " to remote "
            + uploadedFilePath);
      }
      fs.copyFromLocalFile(new Path(fileToUpload), uploadedFilePath);
      uploadedFiles.add(uploadedFilePath);
    }
    return uploadedFilePath;
  }

  public void validFileSize(String uri) throws IOException {
    boolean remote = remoteDirectoryManager.isRemote(uri);
    long actualSizeInBytes = getFileSizeInBytes(uri, remote);
    long maxFileSizeInBytes = convertToBytes(getMaxRemoteFileSizeMB());

    String locationType = remote ? "Remote" : "Local";
    LOG.info("{} file / directory path is {} with size: {} bytes."
        + " Allowed maximum file / directory size is {} bytes.",
        locationType, uri, actualSizeInBytes, maxFileSizeInBytes);

    if (actualSizeInBytes > maxFileSizeInBytes) {
      throw new IOException(
          String.format("Size of file / directory %s: %d bytes. " +
              "This exceeded the configured maximum " +
              "file / directory size, which is %d bytes.",
              uri, actualSizeInBytes, maxFileSizeInBytes));
    }
  }

  private long getFileSizeInBytes(String uri, boolean remote)
      throws IOException {
    long actualSizeInBytes;
    if (remote) {
      actualSizeInBytes = remoteDirectoryManager.getRemoteFileSize(uri);
    } else {
      actualSizeInBytes = FileUtil.getDU(new File(uri));
    }
    return actualSizeInBytes;
  }

  private long getMaxRemoteFileSizeMB() {
    return submarineConfig.getLong(
        SubmarineConfiguration.ConfVars.
            SUBMARINE_LOCALIZATION_MAX_ALLOWED_FILE_SIZE_MB);
  }

  private long convertToBytes(long fileSizeMB) {
    return fileSizeMB * BYTES_IN_MB;
  }

  public void setPermission(Path destPath, FsPermission permission) throws
      IOException {
    FileSystem fs = FileSystem.get(yarnConfig);
    fs.setPermission(destPath, new FsPermission(permission));
  }

  public static boolean needHdfs(List<String> stringsToCheck) {
    for (String content : stringsToCheck) {
      if (content != null && content.contains("hdfs://")) {
        return true;
      }
    }
    return false;
  }

  public static boolean needHdfs(String content) {
    return content != null && content.contains("hdfs://");
  }

  public Set<Path> getUploadedFiles() {
    return uploadedFiles;
  }
}
