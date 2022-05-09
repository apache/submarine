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
package org.apache.submarine.server.database.workbench.utils;

import org.apache.submarine.server.database.workbench.entity.SysDeptSelectEntity;
import org.apache.submarine.server.database.workbench.entity.SysDeptTree;
import org.apache.submarine.server.database.workbench.entity.SysDeptEntity;

import java.util.ArrayList;
import java.util.List;

public class DepartmentUtil {
  public static void disableTargetDeptCode(List<SysDeptSelectEntity> sysDeptSelects,
                                          String deptCode) {
    if (sysDeptSelects == null) {
      return;
    }

    for (SysDeptSelectEntity deptSelect : sysDeptSelects) {
      if (deptSelect.getKey().equalsIgnoreCase(deptCode)) {
        deptSelect.setDisabled(true);
      }
      disableTargetDeptCode(deptSelect.getChildren(), deptCode);
    }
  }

  public static List<SysDeptTree> wrapDeptListToTree(List<SysDeptEntity> sysDeptList,
                                                     List<SysDeptSelectEntity> sysDeptSelects) {
    sysDeptSelects.clear();
    List<SysDeptTree> records = new ArrayList<>();
    for (SysDeptEntity dept : sysDeptList) {
      records.add(new SysDeptTree(dept));
    }
    List<SysDeptTree> sysOrgTreeList = findChildren(records, sysDeptSelects);
    setEmptyChildrenAsNull(sysOrgTreeList);

    return sysOrgTreeList;
  }

  public static long getDeptTreeSize(List<SysDeptTree> sysDeptTreeList) {
    if (sysDeptTreeList == null) {
      return 0;
    }

    long size = 0;
    for (SysDeptTree sysDeptTree : sysDeptTreeList) {
      size += 1 + getDeptTreeSize(sysDeptTree.getChildren());
    }

    return size;
  }

  // Find and encapsulate the node of the top parent class to the TreeList collection
  private static List<SysDeptTree> findChildren(List<SysDeptTree> sysDeptList,
                                                List<SysDeptSelectEntity> sysDeptSelects) {
    List<SysDeptTree> treeList = new ArrayList<>();
    for (SysDeptTree branch : sysDeptList) {
      if (isEmpty(branch.getParentCode())) {
        treeList.add(branch);
        SysDeptSelectEntity departIdModel = new SysDeptSelectEntity().convert(branch);
        sysDeptSelects.add(departIdModel);
      }
    }
    getGrandChildren(treeList, sysDeptList, sysDeptSelects);
    return treeList;
  }

  // Find all child node collections under the top parent class and wrap them in a TreeList collection
  private static void getGrandChildren(List<SysDeptTree> treeList,
                                       List<SysDeptTree> recordList,
                                       List<SysDeptSelectEntity> sysDeptSelects) {
    for (int i = 0; i < treeList.size(); i++) {
      SysDeptTree model = treeList.get(i);
      SysDeptSelectEntity idModel = sysDeptSelects.get(i);
      for (SysDeptTree m : recordList) {
        if (m.getParentCode() != null && m.getParentCode().equals(model.getDeptCode())) {
          model.getChildren().add(m);
          SysDeptSelectEntity dim = new SysDeptSelectEntity().convert(m);
          idModel.getChildren().add(dim);
        }
      }
      getGrandChildren(treeList.get(i).getChildren(), recordList, sysDeptSelects.get(i).getChildren());
    }
  }

  private static void setEmptyChildrenAsNull(List<SysDeptTree> treeList) {
    for (SysDeptTree model : treeList) {
      if (model.getChildren().size() == 0) {
        model.setChildren(null);
      } else {
        setEmptyChildrenAsNull(model.getChildren());
      }
    }
  }

  private static boolean isEmpty(Object object) {
    if (object == null) {
      return true;
    }
    if ("".equals(object)) {
      return true;
    }
    if ("null".equals(object)) {
      return true;
    }
    return false;
  }
}
