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

import java.io.Serializable;
import java.util.Date;

public abstract class BaseEntity implements Serializable {
  private static final long serialVersionUID = 1L;

  protected String id;

  protected String create_by;

  protected Date create_time;

  protected String update_by;

  protected Date update_time;

  public String getId() {
    return id;
  }

  public void setId(String id) {
    this.id = id;
  }

  public String getCreateBy() {
    return create_by;
  }

  public void setCreateBy(String userId) {
    this.create_by = userId;
  }

  public Date getCreateTime() {
    return create_time;
  }

  public void setCreateTime(Date createTime) {
    this.create_time = createTime;
  }

  public String getUpdateBy() {
    return update_by;
  }

  public void setUpdateBy(String userId) {
    this.update_by = userId;
  }

  public Date getUpdateTime() {
    return update_time;
  }

  public void setUpdateTime(Date updateTime) {
    this.update_time = updateTime;
  }
}
