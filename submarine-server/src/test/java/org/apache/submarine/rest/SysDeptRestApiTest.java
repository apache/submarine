/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.rest;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.SysDept;
import org.apache.submarine.database.entity.SysDeptTree;
import org.apache.submarine.server.JsonResponse;
import org.junit.After;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.submarine.rest.SysDeptRestApi.SHOW_ALERT;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class SysDeptRestApiTest {
  private static SysDeptRestApi sysDeptRestApi = new SysDeptRestApi();

  private static final Gson gson = new Gson();

  @After
  public void removeAllTestRecord() {
    // clean department depends
    Response response = sysDeptRestApi.resetParentDept();
    assertResponseSuccess(response);

    // remove all test record
    JsonResponse<QueryResult<SysDeptTree>> response2 = queryDeptTreeList();
    assertTrue(response2.getSuccess());
    for (SysDeptTree deptTree : response2.getResult().getRecords()) {
      Response response3 = sysDeptRestApi.remove(deptTree.getId());
      assertResponseSuccess(response3);
    }

    // Check if all are deleted
    JsonResponse<QueryResult<SysDeptTree>> response4 = queryDeptTreeList();
    assertTrue(response4.getSuccess());
    assertEquals(response4.getResult().getRecords().size(), 0);
    assertEquals(response4.getResult().getTotal(), 0);
  }

  @Test
  public void correctDeptDepend() {
    // Correct department dependencies
    SysDept deptA = new SysDept("A", "deptA");
    SysDept deptAA = new SysDept("AA", "deptAA");
    deptAA.setParentCode("A");
    SysDept deptAB = new SysDept("AB", "deptAB");
    deptAB.setParentCode("A");
    SysDept deptAAA = new SysDept("AAA", "deptAAA");
    deptAAA.setParentCode("AA");
    SysDept deptABA = new SysDept("ABA", "deptABA");
    deptABA.setParentCode("AB");

    List<SysDept> depts = new ArrayList<>();
    depts.addAll(Arrays.asList(deptA, deptAA, deptAB, deptAAA, deptABA));

    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.add(dept);
      assertResponseSuccess(response);
    }

    JsonResponse<QueryResult<SysDeptTree>> response = queryDeptTreeList();
    assertEquals(response.getAttributes().size(), 0);
    assertEquals(response.getResult().getTotal(), 5);
  }

  @Test
  public void errorDeptDepend() {
    // error department dependencies
    SysDept deptA = new SysDept("A", "deptA");
    SysDept deptAA = new SysDept("AA", "deptAA");
    deptAA.setParentCode("A");
    SysDept deptAB = new SysDept("AB", "deptAB");
    deptAB.setParentCode("A");
    SysDept deptAAA = new SysDept("AAA", "deptAAA");
    deptAAA.setParentCode("AA");
    SysDept deptABA = new SysDept("ABA", "deptABA");
    deptABA.setParentCode("AB");

    List<SysDept> depts = new ArrayList<>();
    depts.addAll(Arrays.asList(deptA, deptAA, deptAB, deptAAA, deptABA));
    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.add(dept);
      assertResponseSuccess(response);
    }

    // update error depend
    deptA.setParentCode("AA");
    Response response = sysDeptRestApi.edit(deptA);
    assertResponseSuccess(response);

    JsonResponse<QueryResult<SysDeptTree>> response2 = queryDeptTreeList();
    assertTrue(response2.getSuccess());
    assertEquals(response2.getAttributes().size(), 1);
    assertEquals(response2.getAttributes().get(SHOW_ALERT), Boolean.TRUE);
  }

  @Test
  public void editTest() {
    SysDept deptA = new SysDept("A", "deptA");
    Response response = sysDeptRestApi.add(deptA);
    assertResponseSuccess(response);

    // modify
    deptA.setDeptCode("A-modify");
    deptA.setDeptName("deptA-modify");
    deptA.setParentCode("A-modify");
    deptA.setDeleted(5);
    deptA.setDescription("desc");
    deptA.setSortOrder(9);
    response = sysDeptRestApi.edit(deptA);
    assertResponseSuccess(response);

    // check
    JsonResponse<QueryResult<SysDeptTree>> response4 = queryDeptTreeList();
    SysDeptTree sysDeptTree = response4.getResult().getRecords().get(0);
    assertEquals(sysDeptTree.getDeptCode(), "A-modify");
    assertEquals(sysDeptTree.getDeptName(), "deptA-modify");
    assertEquals(sysDeptTree.getParentCode(), "A-modify");
    // NOTE: parent_name value is left join query
    assertEquals(sysDeptTree.getParentName(), "deptA-modify");
    assertTrue(sysDeptTree.getDeleted() == 5);
    assertEquals(sysDeptTree.getDescription(), "desc");
    assertTrue(sysDeptTree.getSortOrder() == 9);
  }

  @Test
  public void resetParentDeptTest() {
    SysDept deptA = new SysDept("A", "deptA");
    SysDept deptAA = new SysDept("AA", "deptAA");
    deptAA.setParentCode("A");
    SysDept deptAB = new SysDept("AB", "deptAB");
    deptAB.setParentCode("A");

    List<SysDept> depts = new ArrayList<>();
    depts.addAll(Arrays.asList(deptA, deptAA, deptAB));
    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.add(dept);
      assertResponseSuccess(response);
    }

    Response response = sysDeptRestApi.resetParentDept();
    assertResponseSuccess(response);

    JsonResponse<QueryResult<SysDeptTree>> response2 = queryDeptTreeList();
    assertTrue(response2.getSuccess());
    for (SysDeptTree deptTree : response2.getResult().getRecords()) {
      assertEquals(deptTree.getParentCode(), null);
    }
  }

  @Test
  public void deleteTest() {
    SysDept deptA = new SysDept("A", "deptA");

    List<SysDept> depts = new ArrayList<>();
    depts.addAll(Arrays.asList(deptA));
    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.add(dept);
      assertResponseSuccess(response);
    }

    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.delete(dept.getId(), 1);
      assertResponseSuccess(response);
    }

    JsonResponse<QueryResult<SysDeptTree>> response2 = queryDeptTreeList();
    assertTrue(response2.getSuccess());
    for (SysDeptTree deptTree : response2.getResult().getRecords()) {
      assertTrue(deptTree.getDeleted() == 1);
    }
  }

  @Test
  public void deleteBatchTest() {
    SysDept deptA = new SysDept("A", "deptA");
    SysDept deptAA = new SysDept("AA", "deptAA");
    SysDept deptAB = new SysDept("AB", "deptAB");

    StringBuilder ids = new StringBuilder();
    List<SysDept> depts = new ArrayList<>();
    depts.addAll(Arrays.asList(deptA, deptAA, deptAB));
    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.add(dept);
      assertResponseSuccess(response);
      ids.append(dept.getId() + ",");
    }

    Response response = sysDeptRestApi.deleteBatch(ids.toString());
    assertResponseSuccess(response);

    JsonResponse<QueryResult<SysDeptTree>> response2 = queryDeptTreeList();
    assertTrue(response2.getSuccess());
    for (SysDeptTree deptTree : response2.getResult().getRecords()) {
      assertTrue(deptTree.getDeleted() == 1);
    }
  }

  private JsonResponse<QueryResult<SysDeptTree>> queryDeptTreeList() {
    Response response = sysDeptRestApi.tree(null, null);
    JsonResponse<QueryResult<SysDeptTree>> jsonResponse = wrapResponse(response);

    assertTrue(jsonResponse.getSuccess());
    return jsonResponse;
  }

  private JsonResponse<QueryResult<SysDeptTree>> wrapResponse(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<QueryResult<SysDeptTree>>>() {}.getType();
    JsonResponse<QueryResult<SysDeptTree>> jsonResponse = gson.fromJson(entity, type);

    return jsonResponse;
  }

  private void assertResponseSuccess(Response response) {
    JsonResponse<QueryResult<SysDeptTree>> jsonResponse = wrapResponse(response);
    assertTrue(jsonResponse.getSuccess());
  }
}
