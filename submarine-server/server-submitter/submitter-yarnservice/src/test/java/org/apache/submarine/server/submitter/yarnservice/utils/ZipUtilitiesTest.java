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

import com.google.common.collect.Sets;
import org.apache.submarine.FileUtilitiesForTests;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import static org.junit.Assert.assertEquals;

/**
 * Test class for {@link ZipUtilities}.
 */
public class ZipUtilitiesTest {
  private static final Logger LOG =
      LoggerFactory.getLogger(ZipUtilitiesTest.class);

  private FileUtilitiesForTests fileUtils = new FileUtilitiesForTests();
  private File tempDir;

  @Before
  public void setUp() {
    fileUtils.setup();
    tempDir = fileUtils.createDirInTempDir("testDir");
  }

  @After
  public void teardown() throws IOException {
    fileUtils.teardown();
  }

  private File getDestinationFile() {
    File dstFile = fileUtils.getTempFileWithName(String.format(
        "testFile_%d.zip", System.nanoTime()));
    fileUtils.addTrackedFile(dstFile);

    return dstFile;
  }

  @Test
  public void testZipEmptyDir() throws IOException {
    File dstFile = getDestinationFile();

    ZipUtilities.zipDir(tempDir.getAbsolutePath(), dstFile.getAbsolutePath());
    assertCountOfZipEntries(dstFile, 0);
  }

  @Test
  public void testZipDirWithOneFile() throws IOException {
    fileUtils.createFileInDir(tempDir, "test1");
    File dstFile = getDestinationFile();

    ZipUtilities.zipDir(tempDir.getAbsolutePath(), dstFile.getAbsolutePath());
    assertCountOfZipEntries(dstFile, 1);
  }

  @Test
  public void testZipDirWithMultipleFiles() throws IOException {
    fileUtils.createFileInDir(tempDir, "test1");
    fileUtils.createFileInDir(tempDir, "test2");
    fileUtils.createFileInDir(tempDir, "test3");
    File dstFile = getDestinationFile();

    ZipUtilities.zipDir(tempDir.getAbsolutePath(), dstFile.getAbsolutePath());
    assertCountOfZipEntries(dstFile, 3);
  }

  @Test
  public void testZipDirComplex() throws IOException {
    fileUtils.createFileInDir(tempDir, "test1");
    fileUtils.createFileInDir(tempDir, "test2");
    fileUtils.createFileInDir(tempDir, "test3");
    File subdir1 = fileUtils.createDirectory(tempDir, "subdir1");
    File subdir2 = fileUtils.createDirectory(tempDir, "subdir2");
    fileUtils.createFileInDir(subdir1, "file1_1");
    fileUtils.createFileInDir(subdir1, "file1_2");
    fileUtils.createFileInDir(subdir2, "file2_1");
    fileUtils.createFileInDir(subdir2, "file2_2");

    File dstFile = getDestinationFile();

    ZipUtilities.zipDir(tempDir.getAbsolutePath(), dstFile.getAbsolutePath());
    assertZipEntriesByName(dstFile, Sets.newHashSet(
        "test1", "test2", "test3",
        "subdir1/file1_1", "subdir1/file1_2",
        "subdir2/file2_1", "subdir2/file2_2"
    ));
  }

  private void assertCountOfZipEntries(File file, int expected)
      throws IOException {
    ZipFile zipFile = new ZipFile(file);
    Enumeration<? extends ZipEntry> entries = zipFile.entries();

    int count = 0;
    while (entries.hasMoreElements()) {
      count++;
      ZipEntry zipEntry = entries.nextElement();
      LOG.info("Found zipEntry: " + zipEntry);
    }
    assertEquals(expected, count);
  }

  private void assertZipEntriesByName(File file, Set<String> expectedNames)
      throws IOException {
    ZipFile zipFile = new ZipFile(file);
    Enumeration<? extends ZipEntry> entries = zipFile.entries();

    Set<String> actualEntries = Sets.newHashSet();
    while (entries.hasMoreElements()) {
      ZipEntry zipEntry = entries.nextElement();
      LOG.info("Found zipEntry: " + zipEntry);
      actualEntries.add(zipEntry.getName());
    }

    LOG.info("Expected names of ZipEntries: " + expectedNames);
    LOG.info("Actual names of ZipEntries: " + actualEntries);
    assertEquals(expectedNames, actualEntries);
  }
}
