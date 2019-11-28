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

package org.apache.submarine.client.cli.param.runjob;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.List;

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import com.google.common.collect.ImmutableList;

/**
 * Test class for {@link Localizations}.
 */
public class LocalizationsTest {
  private ParametersHolder createParametersHolder(List<String> paramValues)
      throws YarnException {
    ParametersHolder paramHolder = mock(ParametersHolder.class);
    when(paramHolder.getOptionValues(CliConstants.LOCALIZATION))
        .thenReturn(paramValues);
    return paramHolder;
  }

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Test
  public void testEmptyLocalizations() throws YarnException, ParseException {
    List<String> parameters = ImmutableList.of();
    ParametersHolder parametersHolder = createParametersHolder(parameters);
    Localizations parsedResult = Localizations.parse(parametersHolder);
    assertNotNull("Parsed Localizations result should not be null!",
        parsedResult);

    List<Localization> localizations = parsedResult.getLocalizations();
    assertNotNull("Localizations should not be null!", localizations);
    assertTrue("Localizations should be empty!", localizations.isEmpty());
  }

  @Test
  public void testValidLocalizations() throws Exception {
    List<String> parameters =
        ImmutableList.of(
            "hdfs://remote-file1:/local-filename1:rw",
            "s3a://remote-file2:/local-filename2:rw");
    ParametersHolder parametersHolder = createParametersHolder(parameters);
    Localizations parsedResult = Localizations.parse(parametersHolder);
    assertNotNull("Parsed Localizations result should not be null!",
        parsedResult);

    List<Localization> result = parsedResult.getLocalizations();
    assertNotNull("Localizations should not be null!", result);
    assertEquals("Localizations should not be empty!", 2, result.size());

    Localization localization1 = result.get(0);
    assertEquals("hdfs://remote-file1", localization1.getRemoteUri());
    assertEquals("/local-filename1", localization1.getLocalPath());
    assertEquals("rw", localization1.getMountPermission());

    Localization localization2 = result.get(1);
    assertEquals("s3a://remote-file2", localization2.getRemoteUri());
    assertEquals("/local-filename2", localization2.getLocalPath());
    assertEquals("rw", localization2.getMountPermission());
  }

  @Test
  public void testInvalidLocalizations() throws Exception {
    List<String> parameters = ImmutableList.of(
        "blaaa/local-filename1:rw",
        "s3a://remote-file2:/local-filename2:rw");

    expectedException.expect(ParseException.class);
    expectedException.expectMessage("Invalid local file path");

    ParametersHolder parametersHolder = createParametersHolder(parameters);
    Localizations.parse(parametersHolder);
  }
}
