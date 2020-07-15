package org.apache.submarine.server.rest;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.GenericType;
import javax.ws.rs.core.Response;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class EnvironmentRestApiTest {
  private static EnvironmentRestApi environmentStoreApi;
  private static Environment environment;
  String environmentName = "my-submarine-env";
  String kernelName = "team_default_python_3";
  String dockerImage = "continuumio/anaconda3";
  List<String> kernelChannels = Arrays.asList("defaults", "anaconda");
  List<String> kernelDependencies = Arrays.asList(
          "_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0",
          "anaconda=2020.02=py37_0",
          "anaconda-client=1.7.2=py37_0",
          "anaconda-navigator=1.9.12=py37_0");

  @BeforeClass
  public static void init() {
    SubmarineConfiguration submarineConf = SubmarineConfiguration.getInstance();
    submarineConf.setMetastoreJdbcUrl("jdbc:mysql://127.0.0.1:3306/metastore_test?" +
            "useUnicode=true&amp;" +
            "characterEncoding=UTF-8&amp;" +
            "autoReconnect=true&amp;" +
            "failOverReadOnly=false&amp;" +
            "zeroDateTimeBehavior=convertToNull&amp;" +
            "useSSL=false");
    submarineConf.setMetastoreJdbcUserName("metastore_test");
    submarineConf.setMetastoreJdbcPassword("password_test");
    environmentStoreApi = new EnvironmentRestApi();
  }

  @Before
  public void createAndUpdateEnvironment() {
    KernelSpec kernelSpec = new KernelSpec();
    kernelSpec.setName(kernelName);
    kernelSpec.setChannels(kernelChannels);
    kernelSpec.setDependencies(kernelDependencies);
    EnvironmentSpec environmentSpec = new EnvironmentSpec();
    environmentSpec.setDockerImage(dockerImage);
    environmentSpec.setKernelSpec(kernelSpec);
    environmentSpec.setName("foo");
    // Create Environment
    Response createEnvResponse = environmentStoreApi.createEnvironment(environmentSpec);
    assertEquals(createEnvResponse.getStatus(), Response.Status.OK.getStatusCode());

    // Update Environment
    environmentSpec.setName(environmentName);
    Response updateEnvResponse = environmentStoreApi.updateEnvironment(
            environment.getEnvironmentId().toString(), environmentSpec);
    environment = updateEnvResponse.readEntity(new GenericType<Environment>() {});
    assertEquals(updateEnvResponse.getStatus(), Response.Status.OK.getStatusCode());
  }

  @After
  public void deleteEnvironment() {
    Response jsonResponse = environmentStoreApi.deleteEnvironment(environment.getEnvironmentId().toString());
    assertEquals(jsonResponse.getStatus(), Response.Status.OK.getStatusCode());
  }

  @Test
  public void getEnvironment() {
    Response getEnvResponse = environmentStoreApi.getEnvironment(
            environment.getEnvironmentId().toString());
    Environment responseEntity = getEnvResponse.readEntity(new GenericType<Environment>() {});
    assertEquals(responseEntity.getEnvironmentSpec().getName(), environmentName);
    assertEquals(responseEntity.getEnvironmentSpec().getKernelSpec().getName(), kernelName);
    assertEquals(responseEntity.getEnvironmentSpec().getKernelSpec().getChannels(), kernelChannels);
    assertEquals(responseEntity.getEnvironmentSpec().getKernelSpec().getDependencies(), kernelDependencies);
    assertEquals(responseEntity.getEnvironmentSpec().getDockerImage(), dockerImage);
  }
}
