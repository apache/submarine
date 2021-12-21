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
package org.apache.submarine.server;

import io.swagger.v3.jaxrs2.integration.JaxrsOpenApiContextBuilder;
import io.swagger.v3.oas.integration.SwaggerConfiguration;
import io.swagger.v3.oas.integration.OpenApiConfigurationException;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Bootstrap extends HttpServlet {
  @Override
  public void init(ServletConfig config) throws ServletException {

    OpenAPI oas = new OpenAPI();
    Info info = new Info()
             .title("Submarine API")
             .description("The Submarine REST API allows you to access Submarine resources such as, \n" +
                     "experiments, environments and notebooks. The \n" +
                     "API is hosted under the /v1 path on the Submarine server. For example, \n" +
                     "to list experiments on a server hosted at http://localhost:8080, access\n" +
                     "http://localhost:8080/api/v1/experiment/")
             .termsOfService("http://swagger.io/terms/")
             .contact(new Contact()
             .email("dev@submarine.apache.org"))
             .version("0.7.0-SNAPSHOT")
             .license(new License()
             .name("Apache 2.0")
             .url("http://www.apache.org/licenses/LICENSE-2.0.html"));

    oas.info(info);
    List<Server> servers = new ArrayList<>();
    servers.add(new Server().url("/api"));
    oas.servers(servers);

    SwaggerConfiguration oasConfig = new SwaggerConfiguration()
            .openAPI(oas)
            .resourcePackages(Stream.of("org.apache.submarine.server.rest")
                    .collect(Collectors.toSet()))
            .resourceClasses(Stream.of("org.apache.submarine.server.rest.NotebookRestApi",
                    "org.apache.submarine.server.rest.ExperimentRestApi",
                    "org.apache.submarine.server.rest.EnvironmentRestApi")
                    .collect(Collectors.toSet()));

    try {
      new JaxrsOpenApiContextBuilder()
              .openApiConfiguration(oasConfig)
              .buildContext(true);
    } catch (OpenApiConfigurationException e) {
      throw new ServletException(e.getMessage(), e);
    }
  }
}

