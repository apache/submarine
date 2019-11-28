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

import com.google.common.collect.ImmutableList;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.client.cli.param.Quicklink;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test class for {@link QuickLinks}.
 */
public class QuickLinksTest {
  private ParametersHolder createParametersHolder(List<String> paramValues)
      throws YarnException {
    ParametersHolder paramHolder = mock(ParametersHolder.class);
    when(paramHolder.getOptionValues(CliConstants.QUICKLINK))
        .thenReturn(paramValues);
    return paramHolder;
  }

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Test
  public void testEmptyQuicklinks() throws YarnException, ParseException {
    List<String> links = ImmutableList.of();
    ParametersHolder parametersHolder = createParametersHolder(links);
    QuickLinks result = QuickLinks.parse(parametersHolder);
    assertNotNull("Parsed Quicklink result should not be null!", result);

    List<Quicklink> quickLinks = result.getLinks();
    assertNotNull("Links should not be null!", quickLinks);
    assertTrue("Links should be empty!", quickLinks.isEmpty());
  }

  @Test
  public void testValidQuicklinks() throws Exception {
    List<String> links = ImmutableList.of(
        "Notebook_UI=https://master-0:7070",
        "Notebook_UI_2=https://master-1:7071");
    ParametersHolder parametersHolder = createParametersHolder(links);
    QuickLinks result = QuickLinks.parse(parametersHolder);
    assertNotNull("Parsed Quicklink result should not be null!", result);

    List<Quicklink> quickLinks = result.getLinks();
    assertNotNull("Links should not be null!", quickLinks);
    assertEquals("Links should not be empty!", 2, quickLinks.size());

    Quicklink quicklink1 = quickLinks.get(0);
    assertEquals("Notebook_UI", quicklink1.getLabel());
    assertEquals("master-0", quicklink1.getComponentInstanceName());
    assertEquals("https://", quicklink1.getProtocol());
    assertEquals(7070, quicklink1.getPort());

    Quicklink quicklink2 = quickLinks.get(1);
    assertEquals("Notebook_UI_2", quicklink2.getLabel());
    assertEquals("master-1", quicklink2.getComponentInstanceName());
    assertEquals("https://", quicklink2.getProtocol());
    assertEquals(7071, quicklink2.getPort());
  }

  @Test
  public void testInvalidQuicklinks() throws Exception {
    List<String> links = ImmutableList.of(
        "Notebook_UI=https://master-0:7070",
        "Notebook_UI_2=ftp://master-1:7071");

    expectedException.expect(ParseException.class);
    expectedException
        .expectMessage("Quicklinks should start with http or https");

    ParametersHolder parametersHolder = createParametersHolder(links);
    QuickLinks.parse(parametersHolder);
  }
}
