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

public class SysDictItem extends BaseEntity {

  private String dictId;

  private String itemText;

  private String itemValue;

  private String description;

  private Integer sortOrder;

  private Integer deleted;

  public void setDictId(String dictId) {
    this.dictId = dictId;
  }

  public void setItemText(String itemText) {
    this.itemText = itemText;
  }

  public void setItemValue(String itemValue) {
    this.itemValue = itemValue;
  }

  public void setDescription(String description) {
    this.description = description;
  }

  public void setSortOrder(Integer sortOrder) {
    this.sortOrder = sortOrder;
  }

  public void setDeleted(Integer deleted) {
    this.deleted = deleted;
  }

  public Integer getDeleted() {
    return deleted;
  }
}
