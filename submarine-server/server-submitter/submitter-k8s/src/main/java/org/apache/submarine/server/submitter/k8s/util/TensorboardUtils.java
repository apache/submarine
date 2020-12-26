package org.apache.submarine.server.submitter.k8s.util;

public class TensorboardUtils {
  /*
  Prefix constants
   */
  public static final String PV_PREFIX = "tfboard-pv-";
  public static final String HOST_PREFIX = "/tmp/tfboard-logs/";
  public static final String STORAGE = "1Gi";
  public static final String PVC_PREFIX = "tfboard-pvc-";
  public static final String DEPLOY_PREFIX = "tfboard-";
  public static final String POD_PREFIX = "tfboard-";
  public static final String IMAGE_NAME = "tensorflow/tensorflow:1.11.0";
  public static final String SVC_PREFIX = "tfboard-svc-";
  public static final String INGRESS_PREFIX = "tfboard-ingressroute";
  public static final String PATH_PREFIX = "/tfboard-";
}
