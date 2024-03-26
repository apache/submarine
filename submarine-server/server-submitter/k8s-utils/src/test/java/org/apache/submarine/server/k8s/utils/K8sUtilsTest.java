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

package org.apache.submarine.server.k8s.utils;

import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;

public class K8sUtilsTest {

  private static final Logger LOG = LoggerFactory.getLogger(K8sUtilsTest.class);

  @Test
  public void testCastOffsetDatetime() {
    OffsetDateTime now = OffsetDateTime.now();
    LOG.info("current time: {} will cast to string: {}", now, K8sUtils.castOffsetDatetimeToString(now));
    OffsetDateTime testDate1 = OffsetDateTime.parse("2023-05-23T09:01:12Z");
    Assert.assertEquals("2023-05-23T09:01:12Z", K8sUtils.castOffsetDatetimeToString(testDate1));
    OffsetDateTime testDate2 = OffsetDateTime.parse("2023-05-23T17:01:12.000+08:00",
      DateTimeFormatter.ISO_OFFSET_DATE_TIME);
    Assert.assertEquals("2023-05-23T09:01:12Z", K8sUtils.castOffsetDatetimeToString(testDate2));
  }

}
