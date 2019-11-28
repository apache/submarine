package org.apache.submarine.client.cli;

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// package org.apache.hadoop.yarn.submarine.client.cli;

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.commons.runtime.MockClientContext;
import org.junit.Test;

import java.io.IOException;

/**
 * Test class for {@link RoleResourceParser}.
 */
public class RoleResourceParserTest {

  @Test(expected = ParseException.class)
  public void testSimpleResourceParsingNullResource()
      throws ParseException, YarnException, IOException {
    MockClientContext mockClientContext = new MockClientContext();
    RoleResourceParser roleResourceParser =
        new RoleResourceParser(mockClientContext);

    roleResourceParser.parseResource("resourceKey", null);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testSimpleResourceParsingInvalidResource()
      throws ParseException, YarnException, IOException {
    MockClientContext mockClientContext = new MockClientContext();
    RoleResourceParser roleResourceParser =
        new RoleResourceParser(mockClientContext);

    roleResourceParser.parseResource("resourceKey", "someResource");
  }

  @Test
  public void testSimpleResourceParsingValidResource()
      throws ParseException, YarnException, IOException {
    MockClientContext mockClientContext = new MockClientContext();
    RoleResourceParser roleResourceParser =
        new RoleResourceParser(mockClientContext);

    roleResourceParser.parseResource("resourceKey", "memory=2000,vcores=1");
  }

  @Test(expected = ParseException.class)
  public void testResourceParsingNumberOfInstancesSpecified()
      throws ParseException, YarnException, IOException {
    MockClientContext mockClientContext = new MockClientContext();
    RoleResourceParser roleResourceParser =
        new RoleResourceParser(mockClientContext);

    roleResourceParser.parseResource(2, "resourceKey", null);
  }

  @Test
  public void testResourceParsingNumberOfInstancesSpecified2()
      throws ParseException, YarnException, IOException {
    MockClientContext mockClientContext = new MockClientContext();
    RoleResourceParser roleResourceParser =
        new RoleResourceParser(mockClientContext);

    roleResourceParser.parseResource(0, "resourceKey", null);
  }

}