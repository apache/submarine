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

    Assert.assertEquals(pv.getMetadata().getName(), name);
    Assert.assertEquals(pv.getSpec().getHostPath().getPath(), host_path);
    Assert.assertEquals(pv.getSpec().getCapacity().get("storage"), new Quantity(storage));
  }

  @Test
  public void testParsePersistentVolumeClaim() {
    final String id = "123456789";

    final String name = TensorboardUtils.PVC_PREFIX + id;
    final String volume = TensorboardUtils.PV_PREFIX + id;
    final String storage = TensorboardUtils.STORAGE;

    V1PersistentVolumeClaim pvc = VolumeSpecParser.parsePersistentVolumeClaim(name, volume, storage);

    LOG.info(pvc.toString());
    Assert.assertEquals(pvc.getMetadata().getName(), name);
    Assert.assertEquals(pvc.getSpec().getVolumeName(), volume);
    Assert.assertEquals(pvc.getSpec().getResources().getRequests().get("storage"), new Quantity(storage));
  }
}
