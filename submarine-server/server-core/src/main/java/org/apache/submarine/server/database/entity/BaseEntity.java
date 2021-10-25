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
package org.apache.submarine.server.database.entity;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.google.common.annotations.VisibleForTesting;
import org.apache.submarine.server.workbench.database.utils.CustomJsonDateDeserializer;

import java.lang.reflect.Field;
import java.util.Date;

public abstract class BaseEntity {
  protected String id;

  protected String createBy;

  @JsonDeserialize(using = CustomJsonDateDeserializer.class)
  protected Date createTime = new Date();

  protected String updateBy;

  @JsonDeserialize(using = CustomJsonDateDeserializer.class)
  protected Date updateTime = new Date();

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
    StringBuilder buffer = new StringBuilder();

    Class clazz = getClass();
    String fullName = clazz.getName();
    int position = fullName.lastIndexOf(".");
    String shortName = fullName.substring(position + 1);

    buffer.append(shortName);
    buffer.append(": [");

    Field[] fields = clazz.getDeclaredFields();
    Field.setAccessible(fields, true);
    for (Field field: fields) {
      try {
        buffer.append(field.getName());
        buffer.append("=");
        buffer.append(field.get(this));
        buffer.append(", ");
      } catch (IllegalArgumentException | IllegalAccessException  e) {
        e.printStackTrace();
      }
    }
    buffer.setLength(buffer.length() - 2);
    buffer.append("]");

    return buffer.toString();
  }
}
