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

package org.apache.submarine.server.database.initialization;

import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.flywaydb.core.Flyway;
import org.flywaydb.core.api.configuration.FluentConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DatabaseMetadataService {

  private static final Logger LOG = LoggerFactory.getLogger(DatabaseMetadataService.class);

  /**
   * init database metadata by flyway
   */
  public void initDatabaseMetadata() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    if (!conf.getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_METADATA_INIT)) {
      LOG.info("Skip submarine metadata initialization. If you want to init database metadata, " +
          "you can set `submarine.metadata.init` or SUBMARINE_METADATA_INIT env to true.");
      return;
    }

    // get database connection conf
    String jdbcUrl = conf.getJdbcUrl();
    String jdbcUserName = conf.getJdbcUserName();
    String jdbcPassword = conf.getJdbcPassword();

    // flyway config
    FluentConfiguration fluentConfiguration = new FluentConfiguration()
        .dataSource(jdbcUrl, jdbcUserName, jdbcPassword);

    // sql files location
    fluentConfiguration.locations(conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_METADATA_LOCATION));
    // schema metadata history table, control schema version. version id is default release version
    fluentConfiguration.table("submarine_schema_history");
    // if schema metadata history table is missed，try to create it
    fluentConfiguration.baselineOnMigrate(true);
    // Whether to automatically call validate or not when running migrate.
    // If encounter script tuning, we can set this value to false
    fluentConfiguration.validateOnMigrate(
        conf.getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_METADATA_VALIDATE));
    // basic version, we may start it from 0.7.0
    // We can replace this value if we want to update from an intermediate version in some cases
    fluentConfiguration.baselineVersion(
        conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_METADATA_VERSION));

    Flyway flyway = fluentConfiguration.load();
    try {
      flyway.migrate();
    } catch (Exception e) {
      LOG.warn("Error during database initialization. You may need to manually initialize " +
          "the actual metadata and data.", e);
    }
  }

}
