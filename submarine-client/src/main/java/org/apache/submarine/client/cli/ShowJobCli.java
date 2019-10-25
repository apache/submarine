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

package org.apache.submarine.client.cli;

import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.client.cli.param.ShowJobParameters;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.utils.exception.SubmarineException;
import org.apache.submarine.commons.runtime.fs.StorageKeyConstants;
import org.apache.submarine.commons.runtime.fs.SubmarineStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;

public class ShowJobCli extends AbstractCli {
  private static final Logger LOG = LoggerFactory.getLogger(ShowJobCli.class);

  private Options options;
  private ParametersHolder parametersHolder;

  public ShowJobCli(ClientContext cliContext) {
    super(cliContext);
    options = generateOptions();
  }

  public void printUsages() {
    new HelpFormatter().printHelp("job show", options);
  }

  private Options generateOptions() {
    Options options = new Options();
    options.addOption(CliConstants.NAME, true, "Name of the job");
    options.addOption("h", "help", false, "Print help");
    return options;
  }

  private void parseCommandLineAndGetShowJobParameters(String[] args)
      throws IOException, YarnException {
    // Do parsing
    GnuParser parser = new GnuParser();
    CommandLine cli;
    try {
      cli = parser.parse(options, args);
      parametersHolder = ParametersHolder
          .createWithCmdLine(cli, Command.SHOW_JOB);
      parametersHolder.updateParameters(clientContext);
    } catch (ParseException e) {
      printUsages();
    }
  }

  private void printIfNotNull(String keyForPrint, String keyInStorage,
      Map<String, String> jobInfo) {
    if (jobInfo.containsKey(keyInStorage)) {
      System.out.println("\t" + keyForPrint + ": " + jobInfo.get(keyInStorage));
    }
  }

  private void printJobInfo(Map<String, String> jobInfo) {
    System.out.println("Job Meta Info:");
    printIfNotNull("Application Id", StorageKeyConstants.APPLICATION_ID,
        jobInfo);
    printIfNotNull("Input Path", StorageKeyConstants.INPUT_PATH, jobInfo);
    printIfNotNull("Saved Model Path", StorageKeyConstants.SAVED_MODEL_PATH,
        jobInfo);
    printIfNotNull("Checkpoint Path", StorageKeyConstants.CHECKPOINT_PATH,
        jobInfo);
    printIfNotNull("Run Parameters", StorageKeyConstants.JOB_RUN_ARGS,
        jobInfo);
  }

  @VisibleForTesting
  protected void getAndPrintJobInfo() throws IOException {
    SubmarineStorage storage =
        clientContext.getRuntimeFactory().getSubmarineStorage();

    Map<String, String> jobInfo = null;
    try {
      jobInfo = storage.getJobInfoByName(getParameters().getName());
    } catch (IOException e) {
      LOG.error("Failed to retrieve job info", e);
      throw e;
    }

    printJobInfo(jobInfo);
  }

  @VisibleForTesting
  public ShowJobParameters getParameters() {
    return (ShowJobParameters) parametersHolder.getParameters();
  }

  @Override
  public int run(String[] args)
      throws ParseException, IOException, YarnException, InterruptedException,
      SubmarineException {
    if (CliUtils.argsForHelp(args)) {
      printUsages();
      return 0;
    }
    parseCommandLineAndGetShowJobParameters(args);
    getAndPrintJobInfo();
    return 0;
  }
}
