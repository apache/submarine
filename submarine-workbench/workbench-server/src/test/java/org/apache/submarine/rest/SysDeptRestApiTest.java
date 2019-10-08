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
package org.apache.submarine.rest;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.submarine.database.entity.SysDept;
import org.apache.submarine.database.entity.SysDeptTree;
import org.apache.submarine.server.JsonResponse;
import org.apache.submarine.server.JsonResponse.ListResult;
import org.junit.After;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.submarine.rest.SysDeptRestApi.SHOW_ALERT;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class SysDeptRestApiTest {
  private static SysDeptRestApi sysDeptRestApi = new SysDeptRestApi();

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @After
  public void removeAllTestRecord() {
    CommonDataTest.clearDeptTable();
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
      CommonDataTest.assertDeptResponseSuccess(response);
    }

    JsonResponse<ListResult<SysDeptTree>> response = CommonDataTest.queryDeptTreeList();
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
      CommonDataTest.assertDeptResponseSuccess(response);
    }

    // update error depend
    deptA.setParentCode("AA");
    Response response = sysDeptRestApi.edit(deptA);
    CommonDataTest.assertDeptResponseSuccess(response);

    // show alert message in Front-End
    JsonResponse<ListResult<SysDeptTree>> response2 = CommonDataTest.queryDeptTreeList();
    assertTrue(response2.getSuccess());
    assertEquals(response2.getAttributes().size(), 1);
    assertEquals(response2.getAttributes().get(SHOW_ALERT), Boolean.TRUE);
  }

  @Test
  public void editTest() {
    SysDept deptA = new SysDept("A", "deptA");
    Response response = sysDeptRestApi.add(deptA);
    CommonDataTest.assertDeptResponseSuccess(response);

    // modify
    deptA.setDeptCode("A-modify");
    deptA.setDeptName("deptA-modify");
    deptA.setParentCode("A-modify");
    deptA.setDeleted(5);
    deptA.setDescription("desc");
    deptA.setSortOrder(9);
    response = sysDeptRestApi.edit(deptA);
    CommonDataTest.assertDeptResponseSuccess(response);

    // check
    JsonResponse<ListResult<SysDeptTree>> response4 = CommonDataTest.queryDeptTreeList();
    SysDeptTree sysDeptTree = response4.getResult().getRecords().get(0);
    assertEquals(sysDeptTree.getDeptCode(), deptA.getDeptCode());
    assertEquals(sysDeptTree.getDeptName(), deptA.getDeptName());
    assertEquals(sysDeptTree.getParentCode(), deptA.getParentCode());
    // NOTE: parent_name value is left join query
    assertEquals(sysDeptTree.getParentName(), "deptA-modify");
    assertEquals(sysDeptTree.getDeleted(), deptA.getDeleted());
    assertEquals(sysDeptTree.getDescription(), deptA.getDescription());
    assertEquals(sysDeptTree.getSortOrder(), deptA.getSortOrder());
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
      CommonDataTest.assertDeptResponseSuccess(response);
    }

    Response response = sysDeptRestApi.resetParentDept();
    CommonDataTest.assertDeptResponseSuccess(response);

    JsonResponse<ListResult<SysDeptTree>> response2 = CommonDataTest.queryDeptTreeList();
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
      CommonDataTest.assertDeptResponseSuccess(response);
    }

    for (SysDept dept : depts) {
      Response response = sysDeptRestApi.delete(dept.getId(), 1);
      CommonDataTest.assertDeptResponseSuccess(response);
    }

    JsonResponse<ListResult<SysDeptTree>> response2 = CommonDataTest.queryDeptTreeList();
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
      CommonDataTest.assertDeptResponseSuccess(response);
      ids.append(dept.getId() + ",");
    }

    Response response = sysDeptRestApi.deleteBatch(ids.toString());
    CommonDataTest.assertDeptResponseSuccess(response);

    JsonResponse<ListResult<SysDeptTree>> response2 = CommonDataTest.queryDeptTreeList();
    assertTrue(response2.getSuccess());
    for (SysDeptTree deptTree : response2.getResult().getRecords()) {
      assertTrue(deptTree.getDeleted() == 1);
    }
  }
}
