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

import com.google.common.collect.Lists;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.CliUtils;
import org.apache.submarine.client.cli.runjob.RoleParameters;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.api.MXNetRole;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;

import java.io.IOException;
import java.util.List;

/**
 * Parameters for MXNet job.
 */
public class MXNetRunJobParameters extends RunJobParameters {
    private RoleParameters psParameters =
            RoleParameters.createEmpty(MXNetRole.PS);

    private RoleParameters schedulerParameters =
            RoleParameters.createEmpty(MXNetRole.SCHEDULER);

    private static final String CANNOT_BE_DEFINED_FOR_MXNET =
            "cannot be defined for MXNet jobs!";

    @Override
    public void updateParameters(Parameter parametersHolder, ClientContext clientContext)
            throws ParseException, IOException, YarnException {

        checkArguments(parametersHolder);
        super.updateParameters(parametersHolder, clientContext);

        String input = parametersHolder.getOptionValue(CliConstants.INPUT_PATH);
        this.workerParameters = generateWorkerParameters(clientContext, parametersHolder, input);
        this.psParameters = getPSParameters(parametersHolder);
        this.schedulerParameters = getSchedulerParameters(parametersHolder);
        this.distributed = determineIfDistributed(workerParameters.getReplicas(),
                psParameters.getReplicas(), schedulerParameters.getReplicas());

        executePostOperations(clientContext);
    }

    @Override
    void executePostOperations(ClientContext clientContext) throws IOException {
        // Set default job dir / saved model dir, etc.
        setDefaultDirs(clientContext);
        replacePatternsInParameters(clientContext);
    }

    @Override
    public List<String> getLaunchCommands() {
        return Lists.newArrayList(getWorkerLaunchCmd(), getPSLaunchCmd(), getSchedulerLaunchCmd());
    }

    private void replacePatternsInParameters(ClientContext clientContext)
            throws IOException {
        if (StringUtils.isNotEmpty(getPSLaunchCmd())) {
            String afterReplace =
                    CliUtils.replacePatternsInLaunchCommand(getPSLaunchCmd(), this,
                            clientContext.getRemoteDirectoryManager());
            setPSLaunchCmd(afterReplace);
        }
        if (StringUtils.isNotEmpty(getWorkerLaunchCmd())) {
            String afterReplace =
                    CliUtils.replacePatternsInLaunchCommand(getWorkerLaunchCmd(), this,
                            clientContext.getRemoteDirectoryManager());
            setWorkerLaunchCmd(afterReplace);
        }
        if (StringUtils.isNotEmpty(getSchedulerLaunchCmd())) {
            String afterReplace =
                    CliUtils.replacePatternsInLaunchCommand(getSchedulerLaunchCmd(), this,
                            clientContext.getRemoteDirectoryManager());
            setSchedulerLaunchCmd(afterReplace);
        }
    }

    private void checkArguments(Parameter parametersHolder)
        throws YarnException, ParseException {
        if (parametersHolder.hasOption(CliConstants.TENSORBOARD)) {
            throw new ParseException(getParamCannotBeDefinedErrorMessage(
                    CliConstants.TENSORBOARD));
        } else if (parametersHolder
                .getOptionValue(CliConstants.TENSORBOARD_RESOURCES) != null) {
            throw new ParseException(getParamCannotBeDefinedErrorMessage(
                    CliConstants.TENSORBOARD_RESOURCES));
        } else if (parametersHolder
                .getOptionValue(CliConstants.TENSORBOARD_DOCKER_IMAGE) != null) {
            throw new ParseException(getParamCannotBeDefinedErrorMessage(
                    CliConstants.TENSORBOARD_DOCKER_IMAGE));
        }
    }

    private boolean determineIfDistributed(int nWorkers, int nPS, int nSchedulers) {
        return nWorkers >= 2 && nPS > 0 && nSchedulers == 1;
    }

    private String getParamCannotBeDefinedErrorMessage(String cliName) {
        return String.format(
                "Parameter '%s' " + CANNOT_BE_DEFINED_FOR_MXNET, cliName);
    }

    private RoleParameters getPSParameters(Parameter parametersHolder)
            throws YarnException, ParseException {
        int nPS = getNumberOfPS(parametersHolder);
        Resource psResource =
                determinePSResource(parametersHolder, nPS);
        String psDockerImage =
                parametersHolder.getOptionValue(CliConstants.PS_DOCKER_IMAGE);
        String psLaunchCommand =
                parametersHolder.getOptionValue(CliConstants.PS_LAUNCH_CMD);
        return new RoleParameters(MXNetRole.PS, nPS, psLaunchCommand,
                psDockerImage, psResource);
    }

    private Resource determinePSResource(Parameter parametersHolder, int nPS)
            throws ParseException, YarnException {
        if (nPS > 0) {
            String psResourceStr =
                    parametersHolder.getOptionValue(CliConstants.PS_RES);
            if (psResourceStr == null) {
                throw new ParseException("--" + CliConstants.PS_RES + " is absent.");
            }
            return ResourceUtils.createResourceFromString(psResourceStr);
        }
        return null;
    }

    public String getPSLaunchCmd() {
        return psParameters.getLaunchCommand();
    }

    public void setPSLaunchCmd(String launchCmd) {
        psParameters.setLaunchCommand(launchCmd);
    }

    private int getNumberOfPS(Parameter parametersHolder) throws YarnException {
        int nPS = 0;
        if (parametersHolder.getOptionValue(CliConstants.N_PS) != null) {
            nPS = Integer.parseInt(parametersHolder.getOptionValue(CliConstants.N_PS));
        }
        return nPS;
    }

    private RoleParameters getSchedulerParameters(Parameter parametersHolder)
            throws YarnException, ParseException {
        int nSchedulers = getNumberOfScheduler(parametersHolder);
        Resource schedulerResource =
                determineSchedulerResource(parametersHolder, nSchedulers);
        String schedulerDockerImage =
                parametersHolder.getOptionValue(CliConstants.SCHEDULER_DOCKER_IMAGE);
        String schedulerLaunchCommand =
                parametersHolder.getOptionValue(CliConstants.SCHEDULER_LAUNCH_CMD);
        return new RoleParameters(MXNetRole.SCHEDULER, nSchedulers, schedulerLaunchCommand,
                schedulerDockerImage, schedulerResource);
    }

    private Resource determineSchedulerResource(Parameter parametersHolder, int nSchedulers)
            throws ParseException, YarnException {
        if (nSchedulers > 0) {
            String schedulerResourceStr = parametersHolder.getOptionValue(CliConstants.SCHEDULER_RES);
            if (schedulerResourceStr == null) {
                throw new ParseException("--" + CliConstants.SCHEDULER_RES + " is absent.");
            }
            return ResourceUtils.createResourceFromString(schedulerResourceStr);
        }
        return null;
    }

    private int getNumberOfScheduler(Parameter parametersHolder) throws ParseException, YarnException {
        int nSchedulers = 0;
        if (parametersHolder.getOptionValue(CliConstants.N_SCHEDULERS) != null) {
            nSchedulers = Integer.parseInt(parametersHolder.getOptionValue(CliConstants.N_SCHEDULERS));
            if (nSchedulers > 1 || nSchedulers < 0) {
                throw new ParseException("--" + CliConstants.N_SCHEDULERS + " should be 1 or 0");
            }
        }
        return nSchedulers;
    }

    public String getSchedulerLaunchCmd() {
        return schedulerParameters.getLaunchCommand();
    }

    public void setSchedulerLaunchCmd(String launchCmd) {
        schedulerParameters.setLaunchCommand(launchCmd);
    }

    public int getNumPS() {
        return psParameters.getReplicas();
    }

    public Resource getPsResource() {
        return psParameters.getResource();
    }

    public String getPsDockerImage() {
        return psParameters.getDockerImage();
    }

    public int getNumSchedulers() {
        return schedulerParameters.getReplicas();
    }

    public Resource getSchedulerResource() {
        return schedulerParameters.getResource();
    }

    public String getSchedulerDockerImage() {
        return schedulerParameters.getDockerImage();
    }
}
