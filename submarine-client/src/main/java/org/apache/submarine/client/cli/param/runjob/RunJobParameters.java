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

package org.apache.submarine.client.cli.param.runjob;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CaseFormat;
import com.google.common.collect.Lists;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.RoleResourceParser;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.client.cli.param.Quicklink;
import org.apache.submarine.client.cli.param.RunParameters;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.runtime.api.TensorFlowRole;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;
import org.yaml.snakeyaml.introspector.Property;
import org.yaml.snakeyaml.introspector.PropertyUtils;

import java.beans.IntrospectionException;
import java.io.IOException;
import java.util.List;

/**
 * Parameters used to run a job.
 */
public abstract class RunJobParameters extends RunParameters {

  protected String inputPath;
  private String checkpointPath;
  private QuickLinks quicklinks;
  private Localizations localizations;
  private SecurityParameters securityParameters;

  private boolean waitJobFinish = false;
  private boolean securityDisabled = false;
  private List<String> confPairs = Lists.newArrayList();

  protected final RoleResourceParser roleResourceParser;
  protected RoleParameters workerParameters;
  protected boolean distributed = false;

  public RunJobParameters(RoleResourceParser roleResourceParser) {
    this.roleResourceParser = roleResourceParser;
  }

  @Override
  public void updateParameters(Parameter parametersHolder, ClientContext clientContext)
      throws ParseException, IOException, YarnException {

    String input = parametersHolder.getOptionValue(CliConstants.INPUT_PATH);
    this.checkpointPath = parametersHolder.getOptionValue(
        CliConstants.CHECKPOINT_PATH);

    if (parametersHolder.hasOption(CliConstants.INSECURE_CLUSTER)) {
      setSecurityDisabled(true);
    }

    this.securityParameters = new SecurityParameters(parametersHolder);
    this.waitJobFinish = parametersHolder.hasOption(
        CliConstants.WAIT_JOB_FINISH);
    this.quicklinks = QuickLinks.parse(parametersHolder);
    this.localizations = Localizations.parse(parametersHolder);
    this.confPairs = parametersHolder
        .getOptionValues(CliConstants.ARG_CONF);

    super.updateParameters(parametersHolder, clientContext);
  }

  /**
   * Only check null value. Training job should not ignore INPUT_PATH option,
   * but if nWorkers is 0, INPUT_PATH can be ignored because user can only run
   * Tensorboard.
   *
   * @param parametersHolder {@link ParametersHolder} object.
   * @param nWorkers Number of workers.
   * @return Parsed input path.
   * @throws YarnException If any error occurs while querying the option value
   *           from the {@link ParametersHolder} object.
   * @throws ParseException If input path is not specified and number of worker
   *           instances is not zero.
   */
  protected String parseInputPath(ParametersHolder parametersHolder, int nWorkers)
      throws YarnException, ParseException {
    String input = parametersHolder.getOptionValue(CliConstants.INPUT_PATH);
    if (input == null && nWorkers != 0) {
      throw new ParseException("--" + CliConstants.INPUT_PATH + " is absent");
    }
    return input;
  }

  abstract void executePostOperations(ClientContext clientContext)
      throws IOException;

  /**
   * Sets the checkpoint path.
   * If checkpoint path is not specified along with the CLI arguments,
   * we default to the staging directory of {@link RemoteDirectoryManager}.
   * If saved model path is not specified with the CLI arguments,
   * we are using the same directory as the resolved checkpoint path.
   * @param clientContext {@link ClientContext} object
   * @throws IOException If any problem occurs while querying
   * the checkpoint directory from the {@link RemoteDirectoryManager}.
   */
  void setDefaultDirs(ClientContext clientContext) throws IOException {
    // Create directories if needed
    String jobDir = getCheckpointPath();
    if (jobDir == null) {
      jobDir = getJobDir(clientContext);
      setCheckpointPath(jobDir);
    }

    if (getNumWorkers() > 0) {
      String savedModelDir = getSavedModelPath();
      if (savedModelDir == null) {
        savedModelDir = jobDir;
        setSavedModelPath(savedModelDir);
      }
    }
  }

  private String getJobDir(ClientContext clientContext) throws IOException {
    RemoteDirectoryManager rdm = clientContext.getRemoteDirectoryManager();
    if (getNumWorkers() > 0) {
      return rdm.getJobCheckpointDir(getName(), true).toString();
    } else {
      // when #workers == 0, it means we only launch TB. In that case,
      // point job dir to root dir so all job's metrics will be shown.
      return rdm.getUserRootFolder().toString();
    }
  }

  public abstract List<String> getLaunchCommands();

  public String getInputPath() {
    return inputPath;
  }

  public RunJobParameters setInputPath(String input) {
    this.inputPath = input;
    return this;
  }

  public String getCheckpointPath() {
    return checkpointPath;
  }

  public RunJobParameters setCheckpointPath(String checkpointPath) {
    this.checkpointPath = checkpointPath;
    return this;
  }

  public boolean isWaitJobFinish() {
    return waitJobFinish;
  }

  public List<Quicklink> getQuicklinks() {
    return quicklinks.getLinks();
  }

  public List<Localization> getLocalizations() {
    return localizations.getLocalizations();
  }

  public String getKeytab() {
    return securityParameters.getKeytab();
  }

  public String getPrincipal() {
    return securityParameters.getPrincipal();
  }

  public boolean isSecurityDisabled() {
    return securityDisabled;
  }

  public void setSecurityDisabled(boolean securityDisabled) {
    this.securityDisabled = securityDisabled;
  }

  public boolean isDistributeKeytab() {
    return securityParameters.isDistributeKeytab();
  }

  public List<String> getConfPairs() {
    return confPairs;
  }

  public RunJobParameters setConfPairs(List<String> confPairs) {
    this.confPairs = confPairs;
    return this;
  }

  public void setDistributed(boolean distributed) {
    this.distributed = distributed;
  }

  RoleParameters getWorkerParameters(ClientContext clientContext,
      Parameter parametersHolder, String input)
      throws ParseException, YarnException, IOException {
    int nWorkers = getNumberOfWorkers(parametersHolder, input);
    Resource workerResource =
        determineWorkerResource(parametersHolder, nWorkers, clientContext);
    String workerDockerImage =
        parametersHolder.getOptionValue(CliConstants.WORKER_DOCKER_IMAGE);
    String workerLaunchCmd =
        parametersHolder.getOptionValue(CliConstants.WORKER_LAUNCH_CMD);
    return new RoleParameters(TensorFlowRole.WORKER, nWorkers,
        workerLaunchCmd, workerDockerImage, workerResource);
  }

  private Resource determineWorkerResource(Parameter parametersHolder,
      int nWorkers, ClientContext clientContext)
      throws ParseException, YarnException, IOException {
    if (nWorkers > 0) {
      String workerResourceStr =
          parametersHolder.getOptionValue(CliConstants.WORKER_RES);
      if (workerResourceStr == null) {
        throw new ParseException(
            "--" + CliConstants.WORKER_RES + " is absent.");
      }
      return ResourceUtils.createResourceFromString(workerResourceStr);
    }
    return null;
  }

  private int getNumberOfWorkers(Parameter parametersHolder,
      String input) throws ParseException, YarnException {
    int nWorkers = 1;
    if (parametersHolder.getOptionValue(CliConstants.N_WORKERS) != null) {
      nWorkers = Integer
          .parseInt(parametersHolder.getOptionValue(CliConstants.N_WORKERS));
      // Only check null value.
      // Training job shouldn't ignore INPUT_PATH option
      // But if nWorkers is 0, INPUT_PATH can be ignored because
      // user can only run Tensorboard
      if (null == input && 0 != nWorkers) {
        throw new ParseException(
            "--" + CliConstants.INPUT_PATH + " is absent");
      }
    }
    return nWorkers;
  }

  public String getWorkerLaunchCmd() {
    return workerParameters.getLaunchCommand();
  }

  public void setWorkerLaunchCmd(String launchCmd) {
    workerParameters.setLaunchCommand(launchCmd);
  }

  public int getNumWorkers() {
    return workerParameters.getReplicas();
  }

  public void setNumWorkers(int numWorkers) {
    workerParameters.setReplicas(numWorkers);
  }

  public Resource getWorkerResource() {
    return workerParameters.getResource();
  }

  public void setWorkerResource(Resource resource) {
    workerParameters.setResource(resource);
  }

  public String getWorkerDockerImage() {
    return workerParameters.getDockerImage();
  }

  public void setWorkerDockerImage(String image) {
    workerParameters.setDockerImage(image);
  }

  public boolean isDistributed() {
    return distributed;
  }

  @VisibleForTesting
  public void setSecurityParameters(SecurityParameters securityParameters) {
    this.securityParameters = securityParameters;
  }

  @VisibleForTesting
  public void setLocalizations(Localizations localizations) {
    this.localizations = localizations;
  }

  @VisibleForTesting
  public void setQuicklinks(QuickLinks quicklinks) {
    this.quicklinks = quicklinks;
  }

  @VisibleForTesting
  public static class UnderscoreConverterPropertyUtils extends PropertyUtils {
    @Override
    public Property getProperty(Class<? extends Object> type, String name) throws IntrospectionException {
      if (name.indexOf('_') > -1) {
        name = convertName(name);
      }
      return super.getProperty(type, name);
    }

    private static String convertName(String name) {
      return CaseFormat.UPPER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, name);
    }
  }
}
