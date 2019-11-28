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

// package org.apache.hadoop.yarn.submarine.client.cli.param.runjob;

import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.RoleResourceParser;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.commons.runtime.MockClientContext;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test class for {@link TensorBoardParameters}.
 */
public class TensorBoardParametersTest {
  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  private RoleResourceParser createResourceParser() {
    return new RoleResourceParser(new MockClientContext());
  }

  @Test
  public void testNullRoleResourceParserArgument() throws Exception {
    expectedException.expect(NullPointerException.class);
    expectedException.expectMessage("RoleResourceParser must not be null!");

    new TensorBoardParameters(null, mock(ParametersHolder.class));
  }

  @Test
  public void testNullParametersHolderArgument() throws Exception {
    expectedException.expect(NullPointerException.class);
    expectedException.expectMessage("ParametersHolder must not be null!");

    new TensorBoardParameters(createResourceParser(), null);
  }

  @Test
  public void testTensorBoardEnabledWithoutDockerImage() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);

    when(parametersHolder.hasOption(CliConstants.TENSORBOARD)).thenReturn(true);

    TensorBoardParameters tensorBoardParameters =
        new TensorBoardParameters(createResourceParser(), parametersHolder);

    assertTrue("Tensorboard should be enabled!",
        tensorBoardParameters.isEnabled());
    assertNull("Docker image should be null, as it was not specified!",
        tensorBoardParameters.getDockerImage());
    assertEquals("Resource is not the expected (default)!",
        Resource.newInstance(4096, 1), tensorBoardParameters.getResource());
  }

  @Test
  public void testTensorBoardEnabledWithDockerImage() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);

    when(parametersHolder.hasOption(CliConstants.TENSORBOARD)).thenReturn(true);
    String testDockerImage = "testDockerImage";
    when(parametersHolder.getOptionValue(CliConstants.TENSORBOARD_DOCKER_IMAGE))
        .thenReturn(testDockerImage);

    TensorBoardParameters tensorBoardParameters =
        new TensorBoardParameters(createResourceParser(), parametersHolder);

    assertTrue("Tensorboard should be enabled!",
        tensorBoardParameters.isEnabled());
    assertEquals("Docker image should be null, as it was not specified!",
        testDockerImage, tensorBoardParameters.getDockerImage());
    assertEquals("Resource is not the expected (default)!",
        Resource.newInstance(4096, 1), tensorBoardParameters.getResource());
  }

  @Test
  public void testTensorBoardEnabledWithDockerImageAndResource()
      throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);

    when(parametersHolder.hasOption(CliConstants.TENSORBOARD)).thenReturn(true);
    String testDockerImage = "testDockerImage";
    when(parametersHolder.getOptionValue(CliConstants.TENSORBOARD_DOCKER_IMAGE))
        .thenReturn(testDockerImage);
    when(parametersHolder.getOptionValue(CliConstants.TENSORBOARD_RESOURCES))
        .thenReturn("vcores=10,memory=8000M");

    TensorBoardParameters tensorBoardParameters =
        new TensorBoardParameters(createResourceParser(), parametersHolder);

    assertTrue("Tensorboard should be enabled!",
        tensorBoardParameters.isEnabled());
    assertEquals("Docker image is not the expected!", testDockerImage,
        tensorBoardParameters.getDockerImage());
    assertEquals("Resource is not the expected!",
        Resource.newInstance(8000, 10), tensorBoardParameters.getResource());
  }

  @Test
  public void testTensorBoardDisabled() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);

    String testDockerImage = "testDockerImage";
    when(parametersHolder.getOptionValue(CliConstants.TENSORBOARD_DOCKER_IMAGE))
        .thenReturn(testDockerImage);
    when(parametersHolder.getOptionValue(CliConstants.TENSORBOARD_RESOURCES))
        .thenReturn("vcores=10,memory=8000M");

    TensorBoardParameters tensorBoardParameters =
        new TensorBoardParameters(createResourceParser(), parametersHolder);

    assertFalse("Tensorboard should be disabled!",
        tensorBoardParameters.isEnabled());
    assertNull("Docker image should be null!",
        tensorBoardParameters.getDockerImage());
    assertEquals("Resource is not the expected (default)!",
        Resource.newInstance(4096, 1), tensorBoardParameters.getResource());
  }
}
