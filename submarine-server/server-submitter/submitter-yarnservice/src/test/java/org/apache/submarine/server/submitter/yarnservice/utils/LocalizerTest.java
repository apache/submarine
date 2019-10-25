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

import com.google.common.collect.Lists;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.service.api.records.ConfigFile;
import org.apache.hadoop.yarn.service.api.records.ConfigFile.TypeEnum;
import org.apache.hadoop.yarn.service.api.records.Configuration;
import org.apache.hadoop.yarn.service.api.records.Service;
import org.apache.submarine.FileUtilitiesForTests;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.submarine.server.submitter.yarnservice.FileSystemOperations;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class LocalizerTest {

  public static final String DEFAULT_TEMPFILE = "testFile";
  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Mock
  private RemoteDirectoryManager remoteDirectoryManager;

  @Mock
  private FileSystemOperations fsOperations;

  @Mock
  private Service service;

  private FileUtilitiesForTests fileUtils = new FileUtilitiesForTests();

  private void setupService() {
    Configuration conf = new Configuration();
    conf.setFiles(Lists.newArrayList());
    when(service.getConfiguration()).thenReturn(conf);
  }

  private Localizer createLocalizerWithLocalizations(
      Localization... localizations) {
    RunJobParameters parameters = mock(RunJobParameters.class);
    when(parameters.getLocalizations())
        .thenReturn(Lists.newArrayList(localizations));
    return new Localizer(fsOperations, remoteDirectoryManager, parameters);
  }

  @Before
  public void setUp() throws IOException {
    setupService();
    fileUtils.setup();

    when(remoteDirectoryManager.getJobStagingArea(any(), anyBoolean()))
        .thenReturn(new Path("stagingarea"));
  }

  private void testLocalizeExistingRemoteFileInternal() throws IOException {
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.existsRemoteFile(any(Path.class)))
        .thenReturn(true);

    Localization localization = new Localization();
    localization.setLocalPath(".");
    localization.setRemoteUri("hdfs://dummy");
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);
    verify(fsOperations).validFileSize(anyString());
  }

  private String testLocalizeExistingLocalFileInternal(String localPath)
      throws IOException {
    File testFile = fileUtils.createFileInTempDir(DEFAULT_TEMPFILE);

    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(false);
    String remoteUri = testFile.getAbsolutePath();
    when(fsOperations.uploadToRemoteFile(any(Path.class), anyString()))
        .thenReturn(new Path(remoteUri));

    Localization localization = new Localization();
    localization.setLocalPath(localPath);
    localization.setRemoteUri(remoteUri);
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);
    verify(fsOperations).validFileSize(anyString());

    return remoteUri;
  }

  @After
  public void teardown() throws IOException {
    fileUtils.teardown();
  }

  @Test(expected = FileNotFoundException.class)
  public void testLocalizeNotExistingRemoteFile() throws IOException {
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.existsRemoteFile(any(Path.class)))
        .thenReturn(false);

    Localization localization = new Localization();
    localization.setRemoteUri("hdfs://dummy");
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);
  }

  @Test(expected = FileNotFoundException.class)
  public void testLocalizeNotExistingLocalFile() throws IOException {
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(false);

    Localization localization = new Localization();
    localization.setRemoteUri("file://dummy");
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);
  }

  @Test(expected = IOException.class)
  public void testLocalizeExistingRemoteFileInvalidFileSize()
      throws IOException {
    doThrow(IOException.class).when(fsOperations).validFileSize(anyString());
    testLocalizeExistingRemoteFileInternal();
  }

  @Test(expected = IOException.class)
  public void testLocalizeExistingLocalFileInvalidFileSize()
      throws IOException {
    doThrow(IOException.class).when(fsOperations).validFileSize(anyString());
    testLocalizeExistingLocalFileInternal(".");
  }

  @Test
  public void testLocalizeExistingRemoteFile() throws IOException {
    testLocalizeExistingRemoteFileInternal();
    verify(fsOperations, never()).uploadToRemoteFile(any(Path.class),
        anyString());
    verify(fsOperations, never()).deleteFiles(anyString());

    List<ConfigFile> files = service.getConfiguration().getFiles();
    assertEquals(1, files.size());

    ConfigFile configFile = files.get(0);
    assertEquals(TypeEnum.STATIC, configFile.getType());
    assertEquals("hdfs://dummy", configFile.getSrcFile());
  }

  @Test
  public void testLocalizeExistingLocalFile() throws IOException {
    String remoteUri = testLocalizeExistingLocalFileInternal(".");
    verify(fsOperations, never()).deleteFiles(anyString());
    verify(fsOperations).uploadToRemoteFile(any(Path.class), eq(remoteUri));

    List<ConfigFile> files = service.getConfiguration().getFiles();
    assertEquals(1, files.size());

    ConfigFile configFile = files.get(0);
    assertEquals(TypeEnum.STATIC, configFile.getType());
    assertEquals(remoteUri, configFile.getSrcFile());
    assertEquals("testFile", configFile.getDestFile());
  }

  @Test
  public void testLocalizeExistingLocalFile2() throws IOException {
    String remoteUri = testLocalizeExistingLocalFileInternal("./");
    verify(fsOperations, never()).deleteFiles(anyString());
    verify(fsOperations).uploadToRemoteFile(any(Path.class), eq(remoteUri));

    List<ConfigFile> files = service.getConfiguration().getFiles();
    assertEquals(1, files.size());

    ConfigFile configFile = files.get(0);
    assertEquals(TypeEnum.STATIC, configFile.getType());
    assertEquals(remoteUri, configFile.getSrcFile());
    assertEquals("testFile", configFile.getDestFile());
  }

  @Test
  public void testLocalizeExistingLocalFileAbsolute() throws IOException {
    String remoteUri =
        testLocalizeExistingLocalFileInternal("/dummydir/dummyfile");
    verify(fsOperations, never()).deleteFiles(anyString());
    verify(fsOperations).uploadToRemoteFile(any(Path.class), eq(remoteUri));

    List<ConfigFile> files = service.getConfiguration().getFiles();
    assertEquals(1, files.size());

    ConfigFile configFile = files.get(0);
    assertEquals(TypeEnum.STATIC, configFile.getType());
    assertEquals(remoteUri, configFile.getSrcFile());
    assertEquals("dummyfile", configFile.getDestFile());

    assertEquals(1, service.getConfiguration().getEnv().size());
    String dockerMounts = service.getConfiguration().getEnv()
        .get(EnvironmentUtilities.ENV_DOCKER_MOUNTS_FOR_CONTAINER_RUNTIME);
    assertEquals("dummyfile:/dummydir/dummyfile:rw", dockerMounts);
  }

  @Test(expected = IllegalStateException.class)
  public void testLocalizeExistingRemoteDirectoryNoUnderscoreInName()
      throws IOException {
    when(remoteDirectoryManager.isDir(anyString())).thenReturn(true);
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.existsRemoteFile(any(Path.class)))
        .thenReturn(true);
    String remoteUri = "hdfs://remotedir1/remotedir2";
    when(fsOperations.uploadToRemoteFile(any(Path.class), anyString()))
        .thenReturn(new Path(remoteUri));
    when(fsOperations.downloadAndZip(anyString(), anyString(), eq(true)))
        .thenReturn("remotedir2.zip");

    Localization localization = new Localization();
    localization.setLocalPath("remotedir2");
    localization.setRemoteUri(remoteUri);
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);
    verify(fsOperations).validFileSize(anyString());
  }

  @Test
  public void testLocalizeExistingRemoteDirectory() throws IOException {
    when(remoteDirectoryManager.isDir(anyString())).thenReturn(true);
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.existsRemoteFile(any(Path.class)))
        .thenReturn(true);
    String remoteUri = "hdfs://remotedir1/remotedir2";
    when(fsOperations.uploadToRemoteFile(any(Path.class), anyString()))
        .thenReturn(new Path(remoteUri));
    String zipFileName = "remotedir2_221424.zip";
    when(fsOperations.downloadAndZip(anyString(), anyString(), eq(true)))
        .thenReturn(zipFileName);

    Localization localization = new Localization();
    localization.setLocalPath("/remotedir2");
    localization.setRemoteUri(remoteUri);
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);

    verify(fsOperations).validFileSize(anyString());
    verify(fsOperations).deleteFiles(eq(zipFileName));
    verify(fsOperations, never()).uploadToRemoteFile(any(Path.class),
        eq(remoteUri));

    List<ConfigFile> files = service.getConfiguration().getFiles();
    assertEquals(1, files.size());

    ConfigFile configFile = files.get(0);
    assertEquals(TypeEnum.ARCHIVE, configFile.getType());
    assertEquals(remoteUri, configFile.getSrcFile());
    assertEquals("remotedir2", configFile.getDestFile());

    assertEquals(1, service.getConfiguration().getEnv().size());
    String dockerMounts = service.getConfiguration().getEnv()
        .get(EnvironmentUtilities.ENV_DOCKER_MOUNTS_FOR_CONTAINER_RUNTIME);
    assertEquals("remotedir2:/remotedir2:rw", dockerMounts);
  }

  @Test
  public void testLocalizeNonHdfsRemoteUri() throws IOException {
    when(remoteDirectoryManager.isRemote(anyString())).thenReturn(true);
    when(remoteDirectoryManager.existsRemoteFile(any(Path.class)))
        .thenReturn(true);
    String remoteUri = "remote://remotedir1/remotedir2";
    when(fsOperations.uploadToRemoteFile(any(Path.class), anyString()))
        .thenReturn(new Path(remoteUri));
    String downloadedFileName = "remotedir2_221424";
    when(fsOperations.downloadAndZip(anyString(), anyString(), eq(false)))
        .thenReturn(downloadedFileName);

    Localization localization = new Localization();
    localization.setLocalPath("/remotedir2");
    localization.setRemoteUri(remoteUri);
    Localizer localizer = createLocalizerWithLocalizations(localization);

    localizer.handleLocalizations(service);

    verify(fsOperations).validFileSize(anyString());
    verify(fsOperations).deleteFiles(eq(downloadedFileName));
    verify(fsOperations).uploadToRemoteFile(any(Path.class),
        eq(downloadedFileName));

    List<ConfigFile> files = service.getConfiguration().getFiles();
    assertEquals(1, files.size());

    ConfigFile configFile = files.get(0);
    assertEquals(TypeEnum.STATIC, configFile.getType());
    assertEquals(remoteUri, configFile.getSrcFile());
    assertEquals("remotedir2", configFile.getDestFile());

    assertEquals(1, service.getConfiguration().getEnv().size());
    String dockerMounts = service.getConfiguration().getEnv()
        .get(EnvironmentUtilities.ENV_DOCKER_MOUNTS_FOR_CONTAINER_RUNTIME);
    assertEquals("remotedir2:/remotedir2:rw", dockerMounts);
  }
}
