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

/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.apache.submarine.commons.runtime.param;

import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.conf.SubmarineLogs;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;

import java.io.IOException;

/**
 * Base class of all parameters.
 */
public abstract class BaseParameters {
  private String name;

  protected void updateParameters(Parameter parameter,
      ClientContext clientContext)
      throws ParseException, IOException, YarnException {
    String name = parameter.getOptionValue("name");
    if (name == null) {
      throw new ParseException("--name is absent");
    }

    if (parameter.hasOption("verbose")) {
      SubmarineLogs.verboseOn();
    }

    this.setName(name);
  }

  public String getName() {
    return name;
  }

  public BaseParameters setName(String name) {
    this.name = name;
    return this;
  }
}
