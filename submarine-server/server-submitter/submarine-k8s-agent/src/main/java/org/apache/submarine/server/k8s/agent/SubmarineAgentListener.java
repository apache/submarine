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

package org.apache.submarine.server.k8s.agent;

import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import io.javaoperatorsdk.operator.Operator;
import io.javaoperatorsdk.operator.api.config.ControllerConfigurationOverrider;
import io.javaoperatorsdk.operator.api.reconciler.Reconciler;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.k8s.utils.OwnerReferenceConfig;
import org.reflections.Reflections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.takes.facets.fork.FkRegex;
import org.takes.facets.fork.TkFork;
import org.takes.http.Exit;
import org.takes.http.FtBasic;

import java.io.IOException;
import java.time.format.DateTimeFormatter;
import java.util.Set;

/**
 * Submarine agent listener
 * Listen for changes in the associated kubeflow resources and update their status
 */
public class SubmarineAgentListener {

  private static final Logger LOGGER = LoggerFactory.getLogger(SubmarineAgentListener.class);

  public static final DateTimeFormatter DTF = DateTimeFormatter.ISO_DATE_TIME;

  public static void main(String[] args) throws IOException {
    // create kubernetes client
    KubernetesClient client = new KubernetesClientBuilder().build();
    // create operator
    Operator operator = new Operator(client);
    // scan all Reconciler implemented subclasses
    Reflections reflections = new Reflections("org.apache.submarine.server.k8s.agent");
    Set<Class<? extends Reconciler>> reconcilers = reflections.getSubTypesOf(Reconciler.class);
    reconcilers.forEach(reconciler ->
        {
          try {
            LOGGER.info("Register {} ...", reconciler.getName());
            operator.register(reconciler.getDeclaredConstructor().newInstance(),
                ControllerConfigurationOverrider::watchingOnlyCurrentNamespace
            );
          } catch (Exception e) {
            throw new SubmarineRuntimeException("Can not new instance " + reconciler.getName());
          }
        }
    );
    LOGGER.info("Starting agent with SUBMARINE_UID={}", OwnerReferenceConfig.getSubmarineUid());
    // Adds a shutdown hook that automatically calls stop() when the app shuts down.
    operator.installShutdownHook();
    // start operator
    operator.start();
    // Provide a lightweight service to handle health checks
    new FtBasic(
            new TkFork(new FkRegex("/health", "ALL GOOD.")), 8080
    ).start(Exit.NEVER);
  }

}
