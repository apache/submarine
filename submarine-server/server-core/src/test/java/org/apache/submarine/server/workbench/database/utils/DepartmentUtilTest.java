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
package org.apache.submarine.server.workbench.database.utils;

import org.apache.submarine.server.workbench.database.entity.SysDeptEntity;
import org.apache.submarine.server.workbench.database.entity.SysDeptSelectEntity;
import org.apache.submarine.server.workbench.database.entity.SysDeptTree;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;

public class DepartmentUtilTest {
  private static List<SysDeptEntity> sysDeptList = new ArrayList<>();
  private static List<SysDeptSelectEntity> sysDeptSelects = new ArrayList<>();

  @BeforeClass
  public static void init() {
    SysDeptEntity deptA = new SysDeptEntity("A", "deptA");
    SysDeptEntity deptAA = new SysDeptEntity("AA", "deptAA");
    deptAA.setParentCode("A");
    SysDeptEntity deptAB = new SysDeptEntity("AB", "deptAB");
    deptAB.setParentCode("A");
    SysDeptEntity deptAAA = new SysDeptEntity("AAA", "deptAAA");
    deptAAA.setParentCode("AA");
    SysDeptEntity deptABA = new SysDeptEntity("ABA", "deptABA");
    deptABA.setParentCode("AB");
    sysDeptList.addAll(Arrays.asList(deptA, deptAA, deptAB, deptAAA, deptABA));
  }

  @Test
  public void wrapDeptListToTreeTest() {
    List<SysDeptTree> sysDeptTreeList = DepartmentUtil.wrapDeptListToTree(sysDeptList, sysDeptSelects);
    assertEquals(sysDeptTreeList.size(), 1);
    assertEquals(sysDeptTreeList.get(0).getChildren().size(), 2);
    assertEquals(sysDeptTreeList.get(0).getChildren().get(0).getChildren().size(), 1);
    assertEquals(sysDeptTreeList.get(0).getChildren().get(1).getChildren().size(), 1);

    assertEquals(sysDeptSelects.size(), 1);
    assertEquals(sysDeptSelects.get(0).getChildren().size(), 2);
    assertEquals(sysDeptSelects.get(0).getChildren().get(0).getChildren().size(), 1);
    assertEquals(sysDeptSelects.get(0).getChildren().get(1).getChildren().size(), 1);
  }

  @Test
  public void getDeptTreeSizeTest() {
    List<SysDeptTree> sysDeptTreeList = DepartmentUtil.wrapDeptListToTree(sysDeptList, sysDeptSelects);
    long sizeDeptTreeList = DepartmentUtil.getDeptTreeSize(sysDeptTreeList);
    assertEquals(sizeDeptTreeList, 5);
  }

  @Test
  public void disableTargetDeptCodeTest() {
    DepartmentUtil.disableTargetDeptCode(sysDeptSelects, "AAA");
    assertEquals(sysDeptSelects.get(0).getChildren().get(0).getChildren().get(0).getDisabled(), Boolean.TRUE);
  }
}
