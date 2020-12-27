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

package org.apache.submarine.server.submitter.k8s.parser;
import io.kubernetes.client.custom.Quantity;
import io.kubernetes.client.models.V1PersistentVolume;
import io.kubernetes.client.models.V1PersistentVolumeClaim;
import org.apache.submarine.server.submitter.k8s.util.TensorboardUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VolumeSpecParserTest  {
  private static final Logger LOG = LoggerFactory.getLogger(VolumeSpecParserTest.class);

  @Before
  public void before() {

  }

  @Test
  public void testParsePersistentVolume() {
    final String id = "123456789";

    final String name = TensorboardUtils.PV_PREFIX + id;
    final String host_path = TensorboardUtils.HOST_PREFIX + id;
    final String storage = TensorboardUtils.STORAGE;

    V1PersistentVolume pv = VolumeSpecParser.parsePersistentVolume(name, host_path, storage);
    LOG.info(pv.toString());

    Assert.assertEquals(name, pv.getMetadata().getName());
    Assert.assertEquals(host_path, pv.getSpec().getHostPath().getPath());
    Assert.assertEquals(new Quantity(storage), pv.getSpec().getCapacity().get("storage"));
  }

  @Test
  public void testParsePersistentVolumeClaim() {
    final String id = "123456789";

    final String name = TensorboardUtils.PVC_PREFIX + id;
    final String volume = TensorboardUtils.PV_PREFIX + id;
    final String storage = TensorboardUtils.STORAGE;

    V1PersistentVolumeClaim pvc = VolumeSpecParser.parsePersistentVolumeClaim(name, volume, storage);

    LOG.info(pvc.toString());
    Assert.assertEquals(name, pvc.getMetadata().getName());
    Assert.assertEquals(volume, pvc.getSpec().getVolumeName());
    Assert.assertEquals(new Quantity(storage), pvc.getSpec().getResources().getRequests().get("storage"));
  }
}
