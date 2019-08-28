/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.database.entity;

import java.util.ArrayList;
import java.util.List;

// Corresponding to the submarine-web front-end tree table control
public class SysDeptTree extends SysDept {
  // Use the SysDept.id field as the key in the front-end data tree
  private String key;

  public SysDeptTree(SysDept sysOrg) {
    this.key = sysOrg.getDeptCode();

    this.setId(sysOrg.getId());
    this.setDeptCode(sysOrg.getDeptCode());
    this.setDeptName(sysOrg.getDeptName());
    this.setDeleted(sysOrg.getDeleted());
    this.setDescription(sysOrg.getDescription());
    this.setParentCode(sysOrg.getParentCode());
    this.setParentName(sysOrg.getParentName());
    this.setSortOrder(sysOrg.getSortOrder());
    this.setCreateBy(sysOrg.getCreateBy());
    this.setCreateTime(sysOrg.getCreateTime());
    this.setUpdateBy(sysOrg.getUpdateBy());
    this.setUpdateTime(sysOrg.getUpdateTime());
  }

  private List<SysDeptTree> children = new ArrayList<>();

  public List<SysDeptTree> getChildren() {
    return children;
  }

  public void setChildren(List<SysDeptTree> children) {
    this.children = children;
  }

  public void setKey(String key) {
    this.key = key;
  }

}
