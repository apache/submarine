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

import com.google.common.collect.ImmutableMap;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.RoleResourceParser;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.commons.runtime.MockClientContext;
import org.apache.submarine.commons.runtime.api.PyTorchRole;
import org.apache.submarine.commons.runtime.api.Role;
import org.apache.submarine.commons.runtime.api.TensorFlowRole;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test class for {@link RoleParameters}.
 */
@RunWith(Parameterized.class)
public class RoleParametersTest {

  @Rule
  public ExpectedException expectedException = ExpectedException.none();
  private Role role;

  private Map<Role, Integer> expectedRoleInstanceCounts =
      ImmutableMap.of(
          TensorFlowRole.WORKER, 1,
          TensorFlowRole.PS, 0,
          PyTorchRole.WORKER, 1);

  private Map<Role, String> numberOfInstancesKeys = ImmutableMap.of(
      TensorFlowRole.WORKER, CliConstants.N_WORKERS,
      TensorFlowRole.PS, CliConstants.N_PS,
      PyTorchRole.WORKER, CliConstants.N_WORKERS);

  private Map<Role, String> resourceKeys = ImmutableMap.of(
      TensorFlowRole.WORKER, CliConstants.WORKER_RES,
      TensorFlowRole.PS, CliConstants.PS_RES,
      PyTorchRole.WORKER, CliConstants.WORKER_RES);

  private Map<Role, String> dockerImageKeys =
      ImmutableMap.of(
          TensorFlowRole.WORKER, CliConstants.WORKER_DOCKER_IMAGE,
          TensorFlowRole.PS, CliConstants.PS_DOCKER_IMAGE,
          PyTorchRole.WORKER, CliConstants.WORKER_DOCKER_IMAGE);

  private Map<Role, String> launchCommandKeys =
      ImmutableMap.of(
          TensorFlowRole.WORKER, CliConstants.WORKER_LAUNCH_CMD,
          TensorFlowRole.PS, CliConstants.PS_LAUNCH_CMD,
          PyTorchRole.WORKER, CliConstants.WORKER_LAUNCH_CMD);

  @Parameterized.Parameters(name = "{0}")
  public static Collection<Object[]> getParameters() {
    return Stream
        .of(TensorFlowRole.WORKER, TensorFlowRole.PS, PyTorchRole.WORKER)
        .map(type -> new Object[]{type}).collect(Collectors.toList());
  }

  public RoleParametersTest(Role role) {
    this.role = role;
  }

  private void createDefaultParametersHoldr(ParametersHolder parametersHolder)
      throws YarnException {
    when(parametersHolder.getOptionValue(numberOfInstancesKeys.get(role)))
        .thenReturn(String.valueOf(2));
    //we must add a valid Resource to avoid a premature ParseException
    when(parametersHolder.getOptionValue(resourceKeys.get(role)))
        .thenReturn("vcores=10, memory=2048M");
  }

  private RoleResourceParser createResourceParser() {
    return new RoleResourceParser(new MockClientContext());
  }

  @Test
  public void testNullRoleTypeArgument() throws Exception {
    expectedException.expect(NullPointerException.class);
    expectedException.expectMessage("Role must not be null!");

    new RoleParameters(null, createResourceParser(),
        mock(ParametersHolder.class));
  }

  @Test
  public void testNullRoleResourceParserArgument() throws Exception {
    expectedException.expect(NullPointerException.class);
    expectedException.expectMessage("RoleResourceParser must not be null!");

    new RoleParameters(role, null,
        mock(ParametersHolder.class));
  }

  @Test
  public void testNullParametersHolderArgument() throws Exception {
    expectedException.expect(NullPointerException.class);
    expectedException.expectMessage("ParametersHolder must not be null!");

    new RoleParameters(role, createResourceParser(), null);
  }

  @Test
  public void testNumberOfInstancesDefault() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    when(parametersHolder.getOptionValue(resourceKeys.get(role)))
        .thenReturn("vcores=10, memory=2048M");

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);

    //we must add a valid Resource to avoid a premature ParseException

    assertEquals(String.format("Instance count of type %s " +
            "is not the expected!", role),
        (int) expectedRoleInstanceCounts.get(role),
        roleParameters.getReplicas());
  }

  @Test
  public void testNumberOfInstancesSpecified() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    createDefaultParametersHoldr(parametersHolder);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertEquals(String.format("Instance count of type %s " +
            "is not the expected!", role), 2,
        roleParameters.getReplicas());
  }

  @Test
  public void testNumberOfResourcesUnspecified() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    when(parametersHolder.getOptionValue(numberOfInstancesKeys.get(role)))
        .thenReturn(String.valueOf(0));
    when(parametersHolder.getOptionValue(resourceKeys.get(role)))
        .thenReturn(null);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertNull(String.format("Resource of type %s " +
        "should be null!", role), roleParameters.getResource());
  }

  @Test
  public void testNumberOfResourcesInvalidValue() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    when(parametersHolder.getOptionValue(numberOfInstancesKeys.get(role)))
        .thenReturn(String.valueOf(2));
    when(parametersHolder.getOptionValue(resourceKeys.get(role)))
        .thenReturn("bla");

    expectedException.expect(IllegalArgumentException.class);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertNull(String.format("Resource of type %s " +
        "should be null!", role), roleParameters.getResource());
  }

  @Test
  public void testNumberOfResourcesValidValue() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    createDefaultParametersHoldr(parametersHolder);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertEquals(String.format("Resource of type %s " +
            "is not the expected!", role), Resource.newInstance(2048, 10),
        roleParameters.getResource());
  }

  @Test
  public void testDockerImageUnspecified() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    createDefaultParametersHoldr(parametersHolder);

    when(parametersHolder.getOptionValue(dockerImageKeys.get(role)))
        .thenReturn(null);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertNull(String.format("Docker image of type %s " +
            "should be null!", role),
        roleParameters.getDockerImage());
  }


  @Test
  public void testDockerImageSpecified() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    createDefaultParametersHoldr(parametersHolder);

    String dockerImageName = "testDockerImage";
    when(parametersHolder.getOptionValue(dockerImageKeys.get(role)))
        .thenReturn(dockerImageName);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertEquals(String.format("Docker image of type %s " +
            "is not the expected!", role), dockerImageName,
        roleParameters.getDockerImage());
  }

  @Test
  public void testLaunchCommandUnspecified() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    createDefaultParametersHoldr(parametersHolder);

    when(parametersHolder.getOptionValue(launchCommandKeys.get(role)))
        .thenReturn(null);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertNull(String.format("Launch command of type %s " +
            "should be null!", role),
        roleParameters.getLaunchCommand());
  }

  @Test
  public void testLaunchCommandSpecified() throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    createDefaultParametersHoldr(parametersHolder);

    String launchCommand = "testLaunchCommand";
    when(parametersHolder.getOptionValue(launchCommandKeys.get(role)))
        .thenReturn(launchCommand);

    RoleParameters roleParameters = new RoleParameters(role,
        createResourceParser(), parametersHolder);
    assertEquals(String.format("Launch command of type %s " +
            "is not the expected!", role), launchCommand,
        roleParameters.getLaunchCommand());
  }

}
