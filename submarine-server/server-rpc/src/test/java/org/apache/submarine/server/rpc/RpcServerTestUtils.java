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

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RpcServerTestUtils {
  protected static final Logger LOG =
      LoggerFactory.getLogger(RpcServerTestUtils.class);
  static ExecutorService executor;

  public static void startUp(String testClassName) throws Exception {
    LOG.info("Staring test Submarine rpc server for " + testClassName);

    executor = Executors.newSingleThreadExecutor();
    executor.submit(
        () -> {
          try {
            MockRpcServer.main(new String[]{""});
          } catch (Exception e) {
            LOG.error("Exception in Starting submarine rpc server.", e);
            throw new RuntimeException(e);
          }
        });

    long s = System.currentTimeMillis();
    boolean started = false;
    while (System.currentTimeMillis() - s < 1000 * 60 * 3) {  // 3 minutes
      Thread.sleep(2000);
      started = checkIfRpcServerIsRunning();
      if (started == true) {
        break;
      }
    }
    if (started == false) {
      throw new RuntimeException("Can not start Submarine server.");
    }
    LOG.info("Test Submarine rpc server stared.");
  }

  private static boolean checkIfRpcServerIsRunning()  {
    SubmarineConfiguration config = SubmarineConfiguration.getInstance();
    SubmarineRpcClient client = new SubmarineRpcClient(config);
    boolean isRunning = false;
    try {
      isRunning = client.testRpcConnection();
    } catch (InterruptedException e) {
      LOG.error(e.getMessage(), e);
    }
    return isRunning;
  }
}
