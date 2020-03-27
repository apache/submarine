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

package org.apache.submarine.server.submitter;

import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.SubmitterManager;
import org.apache.submarine.server.api.JobSubmitter;
import org.apache.submarine.server.submitter.mock.MockSubmitter;
import org.junit.Assert;
import org.junit.Test;

public class SubmitterManagerTest {
  @Test
  public void testLoadSubmitter() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    conf.setString(SubmarineConfVars.ConfVars.SUBMARINE_SUBMITTERS, "mock");
    conf.setString(SubmarineConfVars.ConfVars.SUBMARINE_SUBMITTERS_CLASS,
        "mock", "org.apache.submarine.server.submitter.mock.MockSubmitter");
    SubmitterManager manager = new SubmitterManager(conf);
    JobSubmitter submitter = manager.getSubmitterByType("mock");
    Assert.assertNotNull(submitter);
    Assert.assertEquals(MockSubmitter.class, submitter.getClass());
  }
}
