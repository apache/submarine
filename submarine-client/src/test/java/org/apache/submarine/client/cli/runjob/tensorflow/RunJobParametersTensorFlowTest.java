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

package org.apache.submarine.client.cli.runjob.tensorflow;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.ResourceNotFoundException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.RoleResourceParser;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.client.cli.param.Quicklink;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.client.cli.param.runjob.TensorFlowRunJobParameters;
import org.apache.submarine.client.cli.runjob.RunJobCliParsingCommonTest;
import org.apache.submarine.commons.runtime.MockClientContext;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * TensorFlow-specific test class for {@link RunJobParameters}.
 */
public class RunJobParametersTensorFlowTest {
  private static final String TEST_NAME = "testName";
  private static final String TEST_INPUT_PATH = "testInputPath";
  private static final Map<String, String> COMMON_PARAMETERS = ImmutableMap
      .<String, String>builder()
      .put(CliConstants.INPUT_PATH, TEST_INPUT_PATH)
      .put(CliConstants.N_PS, "1")
      .put(CliConstants.N_WORKERS, "2")
      .put(CliConstants.NAME, TEST_NAME)
      .build();

  private static final Map<String, String> COMMON_PARAMETERS_ZERO_PS_WORKERS
      = ImmutableMap.<String, String>builder()
      .put(CliConstants.INPUT_PATH, TEST_INPUT_PATH)
      .put(CliConstants.N_PS, "0")
      .put(CliConstants.N_WORKERS, "0")
      .put(CliConstants.NAME, TEST_NAME)
      .build();

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  private void assertCommonParameters(TensorFlowRunJobParameters result) {
    assertEquals(TEST_INPUT_PATH, result.getInputPath());
    assertEquals(TEST_NAME, result.getName());
    assertEquals(1, result.getNumPS());
    assertEquals(2, result.getNumWorkers());
  }

  private void assertCommonParametersZeroPsZeroWorker(
      TensorFlowRunJobParameters result) {
    assertEquals(TEST_INPUT_PATH, result.getInputPath());
    assertEquals(TEST_NAME, result.getName());
    assertEquals(0, result.getNumPS());
    assertEquals(0, result.getNumWorkers());
  }

  private void runTestCase(Map<String, String> stringParams, Runnable before,
                           Consumer<TensorFlowRunJobParameters> after) throws Exception {
    testcaseInternal(stringParams, ImmutableMap.of(), ImmutableList.of(),
        before, after);
  }

  private void runTestCase(Map<String, String> stringParams,
                           List<String> booleanParams, Runnable before,
                           Consumer<TensorFlowRunJobParameters> after) throws Exception {
    testcaseInternal(stringParams, ImmutableMap.of(), booleanParams, before,
        after);
  }

  private void runTestcaseWithStringListParams(Map<String, String> stringParams,
                                               Map<String, List<String>> stringListParams,
                                               Runnable before, Consumer<TensorFlowRunJobParameters> after)
      throws Exception {
    testcaseInternal(stringParams, stringListParams, ImmutableList.of(), before,
        after);
  }

  private void testcaseInternal(Map<String, String> stringParams,
                                Map<String, List<String>> stringListParams, List<String> booleanParams,
                                Runnable before, Consumer<TensorFlowRunJobParameters> after)
      throws Exception {
    ParametersHolder parametersHolder = mock(ParametersHolder.class);
    addStringParametersToHolder(stringParams, parametersHolder);
    addStringListParametersToHolder(stringListParams, parametersHolder);
    addBooleanParametersToHolder(booleanParams, parametersHolder);

    before.run();

    MockClientContext mockClientContext = RunJobCliParsingCommonTest.getMockClientContext();
    RoleResourceParser roleResourceParser =
        new RoleResourceParser(mockClientContext);
    TensorFlowRunJobParameters runJobParameters =
        new TensorFlowRunJobParameters(roleResourceParser);
    runJobParameters.updateParameters(parametersHolder, mockClientContext);

    after.accept(runJobParameters);
  }

  private void addStringParametersToHolder(Map<String, String> params,
                                           ParametersHolder parametersHolder) {
    params.forEach((key, value) -> {
      try {
        when(parametersHolder.getOptionValue(key)).thenReturn(value);
      } catch (YarnException e) {
        throw new RuntimeException(e);
      }
    });
  }

  private void addStringListParametersToHolder(Map<String, List<String>> params,
                                               ParametersHolder parametersHolder) {
    params.forEach((key, value) -> {
      try {
        when(parametersHolder.getOptionValues(key)).thenReturn(value);
      } catch (YarnException e) {
        throw new RuntimeException(e);
      }
    });
  }

  private void addBooleanParametersToHolder(List<String> params,
                                            ParametersHolder parametersHolder) {
    params
        .forEach(key -> when(parametersHolder.hasOption(key)).thenReturn(true));
  }

  @Test
  public void testUnspecifiedInputPathAndWorkersNonZero() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .put(CliConstants.N_WORKERS, "1")
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.NAME, TEST_NAME)
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(ParseException.class);
      expectedException.expectMessage("--input_path is absent");
    }, (result) -> {
    });
  }

  @Test
  public void testOneWorkersOnePs() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .put(CliConstants.INPUT_PATH, "testInputPath")
            .put(CliConstants.NAME, TEST_NAME)
            .put(CliConstants.N_WORKERS, "1")
            .put(CliConstants.N_PS, "1")
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.PS_RES, "vcores=13, memory=2048M")
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(ParseException.class);
      expectedException.expectMessage(
          "Only specified one worker but non-zero PS");
    }, (result) -> {
    });
  }

  @Test
  public void testOneOrMoreWorkerButNoResourceSpecified() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(ParseException.class);
      expectedException.expectMessage("--worker_resources is absent");
    }, (result) -> {
    });
  }

  @Test
  public void testOneOrMorePSButNoResourceSpecified() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(ParseException.class);
      expectedException.expectMessage("--worker_resources is absent");
    }, (result) -> {
    });
  }

  @Test
  public void testOneOrMoreWorkerButInvalidResourceSpecified()
      throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "bla")
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(IllegalArgumentException.class);
      expectedException.expectMessage("\"bla\" is not a valid resource " +
          "type/amount pair");
    }, (result) -> {
    });
  }

  @Test
  public void testOneOrMoreWorkerButInvalidResourceSpecifiedWrongUnit()
      throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "cores=13K")
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(IllegalArgumentException.class);
      expectedException.expectMessage("Acceptable units are M/G or empty");
    }, (result) -> {
    });
  }

  @Test
  public void testOneOrMoreWorkerButInvalidResourceSpecifiedWrongType()
      throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "bla=13M")
            .build();

    runTestCase(stringParams, () -> {
      expectedException.expect(ResourceNotFoundException.class);
      expectedException.expectMessage("Unknown resource: bla");
    }, (result) -> {
    });
  }

  @Test
  public void testOneOrMoreWorkerValidResourceString() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.PS_RES, "vcores=26, memory=4096M")
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .build();

    runTestCase(stringParams, () -> {
    }, (result) -> {
      assertCommonParameters(result);
      assertEquals(Resource.newInstance(4096, 26), result.getPsResource());
      assertEquals(Resource.newInstance(2048, 13), result.getWorkerResource());
    });
  }

  @Test
  public void testTensorboardInvalidResourceSpecified() throws Exception {
    ImmutableList<String> booleanParams =
        ImmutableList.<String>builder()
            .add(CliConstants.TENSORBOARD)
            .build();

    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.PS_RES, "vcores=10, memory=4048M")
            .put(CliConstants.TENSORBOARD_RESOURCES, "bla")
            .build();

    runTestCase(stringParams, booleanParams, () -> {
      expectedException.expect(IllegalArgumentException.class);
      expectedException.expectMessage("\"bla\" is not a valid resource " +
          "type/amount pair");
    }, (result) -> {
    });
  }

  @Test
  public void testTensorboardInvalidResourceSpecifiedWrongUnit()
      throws Exception {
    ImmutableList<String> booleanParams =
        ImmutableList.<String>builder()
            .add(CliConstants.TENSORBOARD)
            .build();

    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.PS_RES, "vcores=10, memory=4048M")
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.TENSORBOARD_RESOURCES, "cores=13K")
            .build();

    runTestCase(stringParams, booleanParams, () -> {
      expectedException.expect(IllegalArgumentException.class);
      expectedException.expectMessage("Acceptable units are M/G or empty");
    }, (result) -> {
    });
  }

  @Test
  public void testTensorboardInvalidResourceSpecifiedWrongType()
      throws Exception {
    ImmutableList<String> booleanParams =
        ImmutableList.<String>builder()
            .add(CliConstants.TENSORBOARD)
            .build();


    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.PS_RES, "vcores=10, memory=4048M")
            .put(CliConstants.TENSORBOARD_RESOURCES, "bla=13M")
            .build();

    runTestCase(stringParams, booleanParams, () -> {
      expectedException.expect(ResourceNotFoundException.class);
      expectedException.expectMessage("Unknown resource: bla");
    }, (result) -> {
    });
  }

  @Test
  public void testTensorboardValidResourceStringAndDockerImage()
      throws Exception {
    ImmutableList<String> booleanParams =
        ImmutableList.<String>builder()
            .add(CliConstants.TENSORBOARD)
            .build();

    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.PS_RES, "vcores=10, memory=4048M")
            .put(CliConstants.TENSORBOARD_RESOURCES, "vcores=20, memory=6048M")
            .put(CliConstants.TENSORBOARD_DOCKER_IMAGE, "testTensorDockerImage")
            .build();

    runTestCase(stringParams, booleanParams, () -> {
    }, (result) -> {
      assertCommonParameters(result);
      assertEquals(Resource.newInstance(2048, 13), result.getWorkerResource());
      assertEquals(Resource.newInstance(4048, 10), result.getPsResource());

      assertTrue(result.isTensorboardEnabled());
      assertEquals(Resource.newInstance(6048, 20),
          result.getTensorboardResource());
      assertEquals("testTensorDockerImage", result.getTensorboardDockerImage());
    });
  }

  @Test
  public void testTensorboardPropertiesOnlyParsedIfTensorboardSwitchedOn()
      throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS)
            .put(CliConstants.WORKER_RES, "vcores=13, memory=2048M")
            .put(CliConstants.PS_RES, "vcores=10, memory=4048M")
            .put(CliConstants.TENSORBOARD_RESOURCES, "vcores=20, memory=6048M")
            .put(CliConstants.TENSORBOARD_DOCKER_IMAGE, "testTensorDockerImage")
            .build();

    runTestCase(stringParams, () -> {
    }, (result) -> {
      assertCommonParameters(result);

      assertEquals(Resource.newInstance(2048, 13), result.getWorkerResource());
      assertEquals(Resource.newInstance(4048, 10), result.getPsResource());

      assertFalse(result.isTensorboardEnabled());
      assertNull(result.getTensorboardResource());
      assertNull(result.getTensorboardDockerImage());
    });
  }

  @Test
  public void testWaitJobFinish() throws Exception {
    List<String> booleanParams = ImmutableList.of(CliConstants.WAIT_JOB_FINISH);
    runTestCase(COMMON_PARAMETERS_ZERO_PS_WORKERS, booleanParams, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertTrue(result.isWaitJobFinish());
    });
  }

  @Test
  public void testDistributeKeytab() throws Exception {
    List<String> booleanParams =
        ImmutableList.of(CliConstants.DISTRIBUTE_KEYTAB);
    runTestCase(COMMON_PARAMETERS_ZERO_PS_WORKERS, booleanParams, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertTrue(result.isDistributeKeytab());
    });
  }

  @Test
  public void testValidQuicklinks() throws Exception {
    ImmutableMap<String, List<String>> stringListParams =
        ImmutableMap.<String, List<String>>builder()
            .put(CliConstants.QUICKLINK, ImmutableList.of(
                "Notebook_UI=https://master-0:7070",
                "Notebook_UI_2=https://master-1:7071"))
            .build();

    runTestcaseWithStringListParams(
        COMMON_PARAMETERS_ZERO_PS_WORKERS,
        stringListParams, () -> {
        },
        (result) -> {
          assertCommonParametersZeroPsZeroWorker(result);

          assertEquals(2, result.getQuicklinks().size());

          Quicklink quicklink1 = result.getQuicklinks().get(0);
          assertEquals("Notebook_UI", quicklink1.getLabel());
          assertEquals("master-0", quicklink1.getComponentInstanceName());
          assertEquals("https://", quicklink1.getProtocol());
          assertEquals(7070, quicklink1.getPort());

          Quicklink quicklink2 = result.getQuicklinks().get(1);
          assertEquals("Notebook_UI_2", quicklink2.getLabel());
          assertEquals("master-1", quicklink2.getComponentInstanceName());
          assertEquals("https://", quicklink2.getProtocol());
          assertEquals(7071, quicklink2.getPort());

        });

  }


  @Test
  public void testInvalidQuicklinks() throws Exception {
    ImmutableMap<String, List<String>> stringListParams =
        ImmutableMap.<String, List<String>>builder()
            .put(CliConstants.QUICKLINK, ImmutableList.of(
                "Notebook_UI=https://master-0:7070",
                "Notebook_UI_2=ftp://master-1:7071"))
            .build();

    runTestcaseWithStringListParams(COMMON_PARAMETERS_ZERO_PS_WORKERS,
        stringListParams, () -> {
          expectedException.expect(ParseException.class);
          expectedException
              .expectMessage("Quicklinks should start with http or https");

        }, (result) -> {

        });

  }


  @Test
  public void testValidLocalizations() throws Exception {
    ImmutableMap<String, List<String>> stringListParams =
        ImmutableMap.<String, List<String>>builder()
            .put(CliConstants.LOCALIZATION, ImmutableList.of(
                "hdfs://remote-file1:/local-filename1:rw",
                "s3a://remote-file2:/local-filename2:rw"))
            .build();

    runTestcaseWithStringListParams(COMMON_PARAMETERS_ZERO_PS_WORKERS,
        stringListParams, () -> {
        },
        (result) -> {
          assertCommonParametersZeroPsZeroWorker(result);

          assertEquals(2, result.getLocalizations().size());

          Localization localization1 = result.getLocalizations().get(0);
          assertEquals("hdfs://remote-file1", localization1.getRemoteUri());
          assertEquals("/local-filename1", localization1.getLocalPath());
          assertEquals("rw", localization1.getMountPermission());

          Localization localization2 = result.getLocalizations().get(1);
          assertEquals("s3a://remote-file2", localization2.getRemoteUri());
          assertEquals("/local-filename2", localization2.getLocalPath());
          assertEquals("rw", localization2.getMountPermission());

        });

  }


  @Test
  public void testInvalidLocalizations() throws Exception {
    ImmutableMap<String, List<String>> stringListParams =
        ImmutableMap.<String, List<String>>builder()
            .put(CliConstants.LOCALIZATION, ImmutableList.of(
                "blaaa/local-filename1:rw",
                "s3a://remote-file2:/local-filename2:rw"))
            .build();

    runTestcaseWithStringListParams(COMMON_PARAMETERS_ZERO_PS_WORKERS,
        stringListParams, () -> {
          expectedException.expect(ParseException.class);
          expectedException.expectMessage("Invalid local file path");

        }, (res) -> {
        });

  }


  @Test
  public void testDockerImagesNotSpecified() throws Exception {
    runTestCase(COMMON_PARAMETERS_ZERO_PS_WORKERS, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertNull(result.getTensorboardDockerImage());
      assertNull(result.getPsDockerImage());
      assertNull(result.getWorkerDockerImage());
      assertNull(result.getDockerImageName());

    });

  }


  @Test
  public void testDockerImageSpecified() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS_ZERO_PS_WORKERS)
            .put(CliConstants.DOCKER_IMAGE, "testDockerImage")
            .put(CliConstants.PS_DOCKER_IMAGE, "psDockerImage")
            .put(CliConstants.WORKER_DOCKER_IMAGE, "workerDockerImage")
            .build();
    runTestCase(stringParams, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertEquals("testDockerImage", result.getDockerImageName());
      assertEquals("psDockerImage", result.getPsDockerImage());
      assertEquals("workerDockerImage", result.getWorkerDockerImage());

    });

  }


  @Test
  public void testLaunchCommandsNotSpecified() throws Exception {
    runTestCase(COMMON_PARAMETERS_ZERO_PS_WORKERS, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertNull(result.getPSLaunchCmd());
      assertNull(result.getWorkerLaunchCmd());

    });

  }


  @Test
  public void testLaunchCommandsSpecified() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS_ZERO_PS_WORKERS)
            .put(CliConstants.PS_LAUNCH_CMD, "psLaunchCmd")
            .put(CliConstants.WORKER_LAUNCH_CMD, "workerLaunchCmd")
            .build();
    runTestCase(stringParams, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertEquals("psLaunchCmd", result.getPSLaunchCmd());
      assertEquals("workerLaunchCmd", result.getWorkerLaunchCmd());

    });

  }


  @Test
  public void testSavedModelPath() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS_ZERO_PS_WORKERS)
            .put(CliConstants.SAVED_MODEL_PATH, "testSavedModelPath")
            .build();
    runTestCase(stringParams, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertEquals("testSavedModelPath", result.getSavedModelPath());

    });

  }


  @Test
  public void testEnvVars() throws Exception {
    ImmutableMap<String, List<String>> stringListParams =
        ImmutableMap.<String, List<String>>builder()
            .put(CliConstants.ENV, ImmutableList.of("env1", "env2"))
            .build();
    runTestcaseWithStringListParams(COMMON_PARAMETERS_ZERO_PS_WORKERS,
        stringListParams, () -> {
        },
        (result) -> {
          assertCommonParametersZeroPsZeroWorker(result);
          assertEquals(2, result.getEnvVars().size());
          assertEquals("env1", result.getEnvVars().get(0));
          assertEquals("env2", result.getEnvVars().get(1));

        });

  }


  @Test
  public void testQueue() throws Exception {
    ImmutableMap<String, String> stringParams =
        ImmutableMap.<String, String>builder()
            .putAll(COMMON_PARAMETERS_ZERO_PS_WORKERS)
            .put(CliConstants.QUEUE, "testQueue")
            .build();
    runTestCase(stringParams, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertEquals("testQueue", result.getQueue());

    });

  }

  @Test
  public void testNameSpecified() throws Exception {
    runTestCase(COMMON_PARAMETERS_ZERO_PS_WORKERS, () -> {
    }, (result) -> {
      assertCommonParametersZeroPsZeroWorker(result);
      assertEquals(TEST_NAME, result.getName());

    });

  }


  @Test
  public void testNameNotSpecified() throws Exception {
    Map<String, String> stringParams =
        Maps.newHashMap(COMMON_PARAMETERS_ZERO_PS_WORKERS);
    stringParams.remove(CliConstants.NAME);

    runTestCase(stringParams, () -> {
      expectedException.expect(ParseException.class);
      expectedException.expectMessage("--name is absent");

    }, (res) -> {
    });

  }
}