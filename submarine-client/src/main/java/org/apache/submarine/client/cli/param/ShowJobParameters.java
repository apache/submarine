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

package org.apache.submarine.client.cli.param;

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.param.BaseParameters;

import java.io.IOException;

public class ShowJobParameters extends BaseParameters {
  public static ShowJobParameters parse(ParametersHolder parametersHolder,
      ClientContext clientContext)
      throws ParseException, YarnException, IOException {
    ShowJobParameters showJobParameters = new ShowJobParameters();
    showJobParameters.updateParameters(parametersHolder, clientContext);
    return showJobParameters;
  }
}
