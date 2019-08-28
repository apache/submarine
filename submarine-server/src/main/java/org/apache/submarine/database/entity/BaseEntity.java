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

import com.google.common.annotations.VisibleForTesting;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.Date;

public abstract class BaseEntity implements Serializable {
  private static final long serialVersionUID = 1L;

  protected String id;

  protected String createBy;

  protected Date createTime;

  protected String updateBy;

  protected Date updateTime;

  public String getId() {
    return id;
  }

  @VisibleForTesting
  public void setId(String id) {
    this.id = id;
  }

  public String getCreateBy() {
    return createBy;
  }

  public void setCreateBy(String userId) {
    this.createBy = userId;
  }

  public Date getCreateTime() {
    return createTime;
  }

  public void setCreateTime(Date createTime) {
    this.createTime = createTime;
  }

  public String getUpdateBy() {
    return updateBy;
  }

  public void setUpdateBy(String userId) {
    this.updateBy = userId;
  }

  public Date getUpdateTime() {
    return updateTime;
  }

  public void setUpdateTime(Date updateTime) {
    this.updateTime = updateTime;
  }

  public String toString() {
    StringBuffer buffer = new StringBuffer();

    Class clazz = getClass();
    String fullName = clazz.getName();
    int position = fullName.lastIndexOf(".");
    String shortName = fullName.substring(position + 1);

    buffer.append(shortName);
    buffer.append(": [");

    Field[] fields = clazz.getDeclaredFields();
    Field.setAccessible(fields, true);
    for (int i = 0; i < fields.length; i++) {
      Field field = fields[i];
      try {
        buffer.append(field.getName());
        buffer.append("=");
        buffer.append(field.get(this));
        buffer.append(", ");
      } catch (IllegalArgumentException e) {
        e.printStackTrace();
      } catch (IllegalAccessException e) {
        e.printStackTrace();
      }
    }
    buffer.setLength(buffer.length() - 2);
    buffer.append("]");

    return buffer.toString();
  }
}
