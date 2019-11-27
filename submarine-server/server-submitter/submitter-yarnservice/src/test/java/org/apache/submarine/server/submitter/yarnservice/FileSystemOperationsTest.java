/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.server.submitter.yarnservice;

import com.google.common.collect.Lists;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.service.api.records.Component;
import org.apache.hadoop.yarn.service.api.records.ConfigFile;
import org.apache.hadoop.yarn.service.api.records.ConfigFile.TypeEnum;
import org.apache.submarine.FileUtilitiesForTests;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.MockClientContext;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.apache.submarine.FileUtilitiesForTests.FILE_SCHEME;
import static org.apache.submarine.client.cli.yarnservice.YarnServiceRunJobCliCommonsTest.DEFAULT_JOB_NAME;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.Silent.class)
public class FileSystemOperationsTest {
  private static final String TARGET_ZIP_FILE = "targetZipFile";
  private static final String TARGET_ZIP_DIR = "targetZipDir";
  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  private FileUtilitiesForTests fileUtils = new FileUtilitiesForTests();
  private FileSystemOperations fileSystemOperations;
  private Path stagingDir;

  private Path getStagingDir(MockClientContext mockClientContext)
      throws IOException {
    return mockClientContext.getRemoteDirectoryManager()
        .getJobStagingArea(DEFAULT_JOB_NAME, true);
  }

  @Before
  public void setUp() throws IOException {
    fileUtils.setup();
    MockClientContext mockClientContext =
        new MockClientContext(DEFAULT_JOB_NAME);
    fileSystemOperations = new FileSystemOperations(mockClientContext);
    stagingDir = getStagingDir(mockClientContext);
    fileUtils.createDirInTempDir(TARGET_ZIP_DIR);
  }

  @After
  public void teardown() throws IOException {
    fileUtils.teardown();
  }

  @Test
  public void testDownloadLocalFileWithSimpleName() throws IOException {
    String resultFile = fileSystemOperations.download("localFile",
        TARGET_ZIP_FILE);
    assertEquals("localFile", resultFile);
  }

  @Test
  public void testDownloadLocalFile() throws IOException {
    String resultFile = fileSystemOperations.download("/tmp/localFile",
        TARGET_ZIP_FILE);
    assertEquals("/tmp/localFile", resultFile);
  }

  @Test
  public void testDownloadRemoteFile() throws IOException {
    String remoteUri = "hdfs:///tmp/remoteFile";
    fileUtils.createFileInDir(stagingDir, remoteUri);

    String resultFilePath = fileSystemOperations.download(remoteUri,
        TARGET_ZIP_FILE);
    File resultFile = new File(resultFilePath);

    assertTrue(resultFile.exists());
    assertEquals(TARGET_ZIP_FILE, resultFile.getName());
  }

  @Test
  public void testDownloadAndZip() throws IOException {
    String remoteDir = "hdfs://remoteDir/";
    fileUtils.createDirInDir(stagingDir, "remoteDir");

    String resultFilePath = fileSystemOperations.downloadAndZip(remoteDir,
        TARGET_ZIP_DIR);
    File resultFile = new File(resultFilePath);

    assertTrue(resultFile.exists());
    assertTrue(
        String.format(
            "Result file name is '%s' and does not start with prefix '%s'",
            resultFile.getName(), TARGET_ZIP_DIR),
        resultFile.getName().startsWith(TARGET_ZIP_DIR));
  }

  @Test(expected = FileNotFoundException.class)
  public void testUploadToRemoteFileNotExistingFile() throws IOException {
    fileSystemOperations.uploadToRemoteFile(stagingDir, "notExisting");
  }

  @Test
  public void testUploadToRemoteFile() throws IOException {
    File testFile = fileUtils.createFileInTempDir("testFile");
    Path path = fileSystemOperations.uploadToRemoteFile(stagingDir,
        testFile.getAbsolutePath());

    File expectedFile = new File(new File(stagingDir.toString()), "testFile");
    assertEquals(expectedFile.getAbsolutePath(), path.toString());

    Set<Path> uploadedFiles = fileSystemOperations.getUploadedFiles();
    assertEquals(1, uploadedFiles.size());
    List<Path> pathList = Lists.newArrayList(uploadedFiles);
    Path storedPath = pathList.get(0);
    assertEquals(path, storedPath);
  }

  @Test
  public void testUploadToRemoteFileMultipleFiles() throws IOException {
    File testFile1 = fileUtils.createFileInTempDir("testFile1");
    File testFile2 = fileUtils.createFileInTempDir("testFile2");
    Path path1 = fileSystemOperations.uploadToRemoteFile(stagingDir,
        testFile1.getAbsolutePath());
    Path path2 = fileSystemOperations.uploadToRemoteFile(stagingDir,
        testFile2.getAbsolutePath());

    File expectedFile1 = new File(new File(stagingDir.toString()), "testFile1");
    File expectedFile2 = new File(new File(stagingDir.toString()), "testFile2");
    assertEquals(expectedFile1.getAbsolutePath(), path1.toString());
    assertEquals(expectedFile2.getAbsolutePath(), path2.toString());

    Set<Path> uploadedFiles = fileSystemOperations.getUploadedFiles();
    assertEquals(2, uploadedFiles.size());
    List<Path> pathList = Lists.newArrayList(uploadedFiles);
    Collections.sort(pathList);
    Path storedPath1 = pathList.get(0);
    Path storedPath2 = pathList.get(1);
    assertEquals(path1, storedPath1);
    assertEquals(path2, storedPath2);
  }

  @Test
  public void testUploadToRemoteFileAndLocalizeMultipleFiles()
      throws IOException {
    Component comp = new Component();

    File testFile1 = fileUtils.createFileInTempDir("testFile1");
    File testFile2 = fileUtils.createFileInTempDir("testFile2");
    fileSystemOperations.uploadToRemoteFileAndLocalizeToContainerWorkDir(
        stagingDir, testFile1.getAbsolutePath(), "testFileDest1", comp);
    fileSystemOperations.uploadToRemoteFileAndLocalizeToContainerWorkDir(
        stagingDir, testFile2.getAbsolutePath(), "testFileDest2", comp);

    List<ConfigFile> files = comp.getConfiguration().getFiles();
    assertEquals(2, files.size());

    ConfigFile configFile1 = files.get(0);
    assertEquals(TypeEnum.STATIC, configFile1.getType());
    assertEquals("testFileDest1", configFile1.getDestFile());
    File expectedTestFile1 =
        new File(new File(stagingDir.toString()), "testFile1");
    assertEquals(FILE_SCHEME + expectedTestFile1.getAbsolutePath(),
        configFile1.getSrcFile());

    ConfigFile configFile2 = files.get(1);
    assertEquals(TypeEnum.STATIC, configFile2.getType());
    assertEquals("testFileDest2", configFile2.getDestFile());
    File expectedTestFile2 =
        new File(new File(stagingDir.toString()), "testFile2");
    assertEquals(FILE_SCHEME + expectedTestFile2.getAbsolutePath(),
        configFile2.getSrcFile());
  }

  @Test
  public void testValidFileSize() throws IOException {
    ClientContext clientContext = mock(ClientContext.class);

    RemoteDirectoryManager remoteDirectoryManager =
        mock(RemoteDirectoryManager.class);
    when(clientContext.getRemoteDirectoryManager())
        .thenReturn(remoteDirectoryManager);
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.getRemoteFileSize(anyString()))
        .thenReturn(20000L);

    SubmarineConfiguration config =
        SubmarineConfiguration.newInstance();
    config.setLong(SubmarineConfiguration.ConfVars
        .SUBMARINE_LOCALIZATION_MAX_ALLOWED_FILE_SIZE_MB, 21L);
    when(clientContext.getSubmarineConfig()).thenReturn(config);

    fileSystemOperations = new FileSystemOperations(clientContext);
  }

  @Test
  public void testValidFileSizeInvalid() throws IOException {
    ClientContext clientContext = mock(ClientContext.class);

    RemoteDirectoryManager remoteDirectoryManager =
        mock(RemoteDirectoryManager.class);
    when(clientContext.getRemoteDirectoryManager())
        .thenReturn(remoteDirectoryManager);
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.getRemoteFileSize(anyString()))
        .thenReturn(20000L);

    SubmarineConfiguration config =
        SubmarineConfiguration.newInstance();
    config.setLong(SubmarineConfiguration.ConfVars
        .SUBMARINE_LOCALIZATION_MAX_ALLOWED_FILE_SIZE_MB, 19L);
    when(clientContext.getSubmarineConfig()).thenReturn(config);

    fileSystemOperations = new FileSystemOperations(clientContext);
  }
}
