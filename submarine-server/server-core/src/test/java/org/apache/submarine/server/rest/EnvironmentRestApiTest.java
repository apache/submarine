package org.apache.submarine.server.rest;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.workbench.database.entity.SysDict;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class EnvironmentRestApiTest {
  private static EnvironmentRestApi environmentStoreApi;
  private static Environment environment;
  private static String environmentName = "my-submarine-env";
  private static String kernelName = "team_default_python_3";
  private static String dockerImage = "continuumio/anaconda3";
  private static List<String> kernelChannels = Arrays.asList("defaults", "anaconda");
  private static List<String> kernelDependencies = Arrays.asList(
          "_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0",
          "anaconda=2020.02=py37_0",
          "anaconda-client=1.7.2=py37_0",
          "anaconda-navigator=1.9.12=py37_0");

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

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
    environment = getEnvironmentFromResponse(createEnvResponse);
    assertEquals(Response.Status.OK.getStatusCode(), createEnvResponse.getStatus());

    // Update Environment
    environmentSpec.setName(environmentName);
    Response updateEnvResponse = environmentStoreApi.updateEnvironment(
            environment.getEnvironmentId().toString(), environmentSpec);
    environment = getEnvironmentFromResponse(updateEnvResponse);
    assertEquals(Response.Status.OK.getStatusCode(), updateEnvResponse.getStatus());
  }

  @After
  public void deleteEnvironment() {
    Response deleteEnvResponse = environmentStoreApi
            .deleteEnvironment(environment.getEnvironmentId().toString());
    assertEquals(Response.Status.OK.getStatusCode(), deleteEnvResponse.getStatus());
  }

  @Test
  public void getEnvironment() {
    Response getEnvResponse = environmentStoreApi.getEnvironment(
            environment.getEnvironmentId().toString());
    Environment responseEntity = getEnvironmentFromResponse(getEnvResponse);
    assertEquals(environmentName, responseEntity.getEnvironmentSpec().getName());
    assertEquals(kernelName, responseEntity.getEnvironmentSpec().getKernelSpec().getName());
    assertEquals(kernelChannels, responseEntity.getEnvironmentSpec().getKernelSpec().getChannels());
    assertEquals(kernelDependencies, responseEntity.getEnvironmentSpec().getKernelSpec().getDependencies());
    assertEquals(dockerImage, responseEntity.getEnvironmentSpec().getDockerImage());
  }

  private Environment getEnvironmentFromResponse(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<Environment>>() {}.getType();
    JsonResponse<Environment> jsonResponse = gson.fromJson(entity, type);
    environment = jsonResponse.getResult();
    return environment;
  }
}
