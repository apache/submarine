/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.server.rpc;

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.remote.RpcRuntimeFactory;
import org.apache.submarine.client.cli.runjob.RunJobCli;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.RuntimeFactory;
import org.apache.submarine.commons.runtime.exception.SubmarineException;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class SubmarineRpcServerTest {
  private static final Logger LOG =
      LoggerFactory.getLogger(SubmarineRpcServerTest.class.getName());

  @BeforeClass
  public static void init() throws Exception {
    RpcServerTestUtils.startUp(
        SubmarineRpcServerTest.class.getSimpleName());
  }

  @Test
  public void testSubmitSubmarineJob() throws InterruptedException,
      SubmarineException, YarnException, ParseException, IOException {
    LOG.info("testSubmitSubmarineJob start.");
    ClientContext clientContext = new ClientContext();
    RuntimeFactory runtimeFactory = new RpcRuntimeFactory(clientContext);
    clientContext.setRuntimeFactory(runtimeFactory);

    String[] moduleArgs = new String []{
        "--name", "tf-job-001",
        "--framework", "tensorflow",
        "--input_path", "",
        "--num_workers", "2",
        "--worker_resources", "memory=1G,vcores=1",
        "--num_ps", "1",
        "--ps_resources", "memory=1G,vcores=1",
        "--worker_launch_cmd", "${WORKER_CMD}",
        "--ps_launch_cmd",
        "myvenv.zip/venv/bin/python mnist_distributed.py " +
          "--steps 2 --data_dir /tmp/data --working_dir /tmp/mode",
        "--insecure"
    };
    new RunJobCli(clientContext).run(moduleArgs);
    LOG.info("testSubmitSubmarineJob done.");
  }

}
