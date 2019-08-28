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
package org.apache.submarine.database.utils;

import org.apache.submarine.database.entity.SysDeptSelect;
import org.apache.submarine.database.entity.SysDeptTree;
import org.apache.submarine.database.entity.SysDept;

import java.util.ArrayList;
import java.util.List;

public class DepartmentUtil {
  public static void disableTagetDeptCode(List<SysDeptSelect> sysDeptSelects,
                                          String deptCode) {
    if (sysDeptSelects == null) {
      return;
    }

    for (SysDeptSelect deptSelect : sysDeptSelects) {
      if (deptSelect.getKey().equalsIgnoreCase(deptCode)) {
        deptSelect.setDisabled(true);
      }
      disableTagetDeptCode(deptSelect.getChildren(), deptCode);
    }
  }

  public static List<SysDeptTree> wrapDeptListToTree(List<SysDept> sysDeptList,
                                                     List<SysDeptSelect> sysDeptSelects) {
    sysDeptSelects.clear();
    List<SysDeptTree> records = new ArrayList<>();
    for (int i = 0; i < sysDeptList.size(); i++) {
      SysDept dept = sysDeptList.get(i);
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
                                                List<SysDeptSelect> sysDeptSelects) {
    List<SysDeptTree> treeList = new ArrayList<>();
    for (int i = 0; i < sysDeptList.size(); i++) {
      SysDeptTree branch = sysDeptList.get(i);
      if (isEmpty(branch.getParentCode())) {
        treeList.add(branch);
        SysDeptSelect departIdModel = new SysDeptSelect().convert(branch);
        sysDeptSelects.add(departIdModel);
      }
    }
    getGrandChildren(treeList, sysDeptList, sysDeptSelects);
    return treeList;
  }

  // Find all child node collections under the top parent class and wrap them in a TreeList collection
  private static void getGrandChildren(List<SysDeptTree> treeList,
                                       List<SysDeptTree> recordList,
                                       List<SysDeptSelect> sysDeptSelects) {
    for (int i = 0; i < treeList.size(); i++) {
      SysDeptTree model = treeList.get(i);
      SysDeptSelect idModel = sysDeptSelects.get(i);
      for (int i1 = 0; i1 < recordList.size(); i1++) {
        SysDeptTree m = recordList.get(i1);
        if (m.getParentCode() != null && m.getParentCode().equals(model.getDeptCode())) {
          model.getChildren().add(m);
          SysDeptSelect dim = new SysDeptSelect().convert(m);
          idModel.getChildren().add(dim);
        }
      }
      getGrandChildren(treeList.get(i).getChildren(), recordList, sysDeptSelects.get(i).getChildren());
    }
  }

  private static void setEmptyChildrenAsNull(List<SysDeptTree> treeList) {
    for (int i = 0; i < treeList.size(); i++) {
      SysDeptTree model = treeList.get(i);
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
