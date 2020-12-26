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
