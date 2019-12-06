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

package org.apache.submarine.server.submitter.yarnservice.utils;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.service.api.records.ConfigFile;
import org.apache.hadoop.yarn.service.api.records.ConfigFile.TypeEnum;
import org.apache.hadoop.yarn.service.api.records.Service;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.submarine.server.submitter.yarnservice.FileSystemOperations;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import static org.apache.submarine.server.submitter.yarnservice.FileSystemOperations.needHdfs;
import static org.apache.submarine.server.submitter.yarnservice.utils.EnvironmentUtilities.appendToEnv;

/**
 * This class holds all dependencies in order to localize dependencies
 * for containers.
 */
public class Localizer {
  private static final Logger LOG = LoggerFactory.getLogger(Localizer.class);

  private final FileSystemOperations fsOperations;
  private final RemoteDirectoryManager remoteDirectoryManager;
  private final RunJobParameters parameters;

  public Localizer(FileSystemOperations fsOperations,
      RemoteDirectoryManager remoteDirectoryManager,
      RunJobParameters parameters) {
    this.fsOperations = fsOperations;
    this.remoteDirectoryManager = remoteDirectoryManager;
    this.parameters = parameters;
  }

  /**
   * Localize dependencies for all containers.
   * If remoteUri is a local directory,
   * we'll zip it, upload to HDFS staging dir HDFS.
   * If remoteUri is directory, we'll download it, zip it and upload
   * to HDFS.
   * If localFilePath is ".", we'll use remoteUri's file/dir name
   */
  public void handleLocalizations(Service service)
      throws IOException {
    // Handle localizations
    Path stagingDir =
        remoteDirectoryManager.getJobStagingArea(
            parameters.getName(), true);
    List<Localization> localizations = parameters.getLocalizations();

    // Check to fail fast
    checkFilesExist(localizations);

    // Start download remote if needed and upload to HDFS
    for (Localization localization : localizations) {
      LocalizationState localizationState = new LocalizationState(localization,
          remoteDirectoryManager);
      Path resourceToLocalize = new Path(localizationState.remoteUri);
      String sourceFile = determineSourceFile(localizationState);

      if (localizationState.needUploadToHDFS) {
        resourceToLocalize =
            fsOperations.uploadToRemoteFile(stagingDir, sourceFile);
      }
      if (localizationState.needToDeleteTempFile) {
        fsOperations.deleteFiles(sourceFile);
      }
      // Remove .zip from zipped dir name
      if (isZippedArchive(sourceFile, localizationState.destFileType)) {
        // Delete local zip file
        fsOperations.deleteFiles(sourceFile);
        sourceFile = getNameUntilUnderscore(sourceFile);
      }

      String containerLocalPath = localizationState.containerLocalPath;
      // If provided, use the name of local uri
      if (!containerLocalPath.equals(".")
          && !containerLocalPath.equals("./")) {
        // Change the YARN localized file name to what will be used in container
        sourceFile = getLastNameFromPath(containerLocalPath);
      }
      String localizedName = getLastNameFromPath(sourceFile);
      LOG.info("The file or directory to be localized is {}. " +
          "Its localized filename will be {}",
          resourceToLocalize.toString(), localizedName);
      ConfigFile configFile = new ConfigFile()
          .srcFile(resourceToLocalize.toUri().toString())
          .destFile(localizedName)
          .type(localizationState.destFileType);
      service.getConfiguration().getFiles().add(configFile);

      if (containerLocalPath.startsWith("/")) {
        addToMounts(service, localization, containerLocalPath, sourceFile);
      }
    }
  }

  private String determineSourceFile(LocalizationState localizationState) throws IOException {
    if (localizationState.directory) {
      // Special handling of remoteUri directory.
      return fsOperations.downloadAndZip(localizationState.remoteUri,
          getLastNameFromPath(localizationState.remoteUri), true);
    } else if (localizationState.remote &&
        !needHdfs(localizationState.remoteUri)) {
      // Non HDFS remote URI.
      // Non directory, we don't need to zip
      return fsOperations.downloadAndZip(localizationState.remoteUri,
          getLastNameFromPath(localizationState.remoteUri), false);
    }
    return localizationState.remoteUri;
  }

  private static String getNameUntilUnderscore(String sourceFile) {
    int suffixIndex = sourceFile.lastIndexOf('_');
    if (suffixIndex == -1) {
      throw new IllegalStateException(String.format(
          "Vale of archive filename"
              + " supposed to contain an underscore. Filename was: '%s'",
          sourceFile));
    }
    sourceFile = sourceFile.substring(0, suffixIndex);
    return sourceFile;
  }

  private static boolean isZippedArchive(String sourceFile,
      TypeEnum destFileType) {
    return destFileType == TypeEnum.ARCHIVE
        && sourceFile.endsWith(".zip");
  }

  // set mounts
  // if mount path is absolute, just use it.
  // if relative, no need to mount explicitly
  private static void addToMounts(Service service, Localization loc,
      String containerLocalPath, String sourceFile) {
    String mountStr = getLastNameFromPath(sourceFile) + ":"
        + containerLocalPath + ":" + loc.getMountPermission();
    LOG.info("Add bind-mount string {}", mountStr);
    appendToEnv(service,
        EnvironmentUtilities.ENV_DOCKER_MOUNTS_FOR_CONTAINER_RUNTIME,
        mountStr, ",");
  }

  private void checkFilesExist(List<Localization> localizations)
      throws IOException {
    String remoteUri;
    for (Localization localization : localizations) {
      remoteUri = localization.getRemoteUri();
      Path resourceToLocalize = new Path(remoteUri);

      if (remoteDirectoryManager.isRemote(remoteUri)) {
        if (!remoteDirectoryManager.existsRemoteFile(resourceToLocalize)) {
          throw new FileNotFoundException(
              "File " + remoteUri + " doesn't exists.");
        }
      } else {
        File localFile = new File(remoteUri);
        if (!localFile.exists()) {
          throw new FileNotFoundException(
              "File " + remoteUri + " doesn't exists.");
        }
      }
      // check remote file size
      fsOperations.validFileSize(remoteUri);
    }
  }

  private enum LocalizationType {
    REMOTE_FILE, REMOTE_DIRECTORY, LOCAL_FILE, LOCAL_DIRECTORY
  }

  private static String getLastNameFromPath(String sourceFile) {
    return new Path(sourceFile).getName();
  }

  private static class LocalizationState {
    private final String remoteUri;
    private final LocalizationType localizationType;
    private final boolean needHdfs;
    private final boolean needUploadToHDFS;
    private final boolean needToDeleteTempFile;
    private final String containerLocalPath;
    private final TypeEnum destFileType;
    private final boolean directory;
    private final boolean remote;

    LocalizationState(Localization localization,
        RemoteDirectoryManager remoteDirectoryManager) throws IOException {
      this.remoteUri = localization.getRemoteUri();
      this.directory = remoteDirectoryManager.isDir(remoteUri);
      this.remote = remoteDirectoryManager.isRemote(remoteUri);
      this.localizationType = determineLocalizationType(directory, remote);
      this.needHdfs = determineNeedHdfs(remote);
      //HDFS file don't need to be uploaded
      this.needUploadToHDFS =
          directory || (remote && !this.needHdfs) || !remote;
      this.needToDeleteTempFile = remote && !this.needHdfs;
      this.containerLocalPath = localization.getLocalPath();
      this.destFileType = directory ? TypeEnum.ARCHIVE : TypeEnum.STATIC;
    }

    private boolean determineNeedHdfs(boolean remote) {
      return remote && needHdfs(remoteUri);
    }

    private LocalizationType determineLocalizationType(boolean directory, boolean remote) {
      if (directory) {
        return remote ? LocalizationType.REMOTE_DIRECTORY : LocalizationType.LOCAL_DIRECTORY;
      } else {
        return remote ? LocalizationType.REMOTE_FILE : LocalizationType.LOCAL_FILE;
      }
    }
  }
}
