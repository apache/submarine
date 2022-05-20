/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.submarine.server.websocket;

import com.google.gson.Gson;
import org.slf4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class Message {
  /**
   * Representation of event type.
   */
  public enum OP {
    ERROR_INFO,                   // [s-c] error information to be sent
    NOTICE                        // [s-c] Notice
  }

  private static final Gson gson = new Gson();
  public static final Message EMPTY = new Message(null);

  public OP op;
  public Map<String, Object> data = new HashMap<>();

  public Message(OP op) {
    this.op = op;
  }

  public Message put(String k, Object v) {
    data.put(k, v);
    return this;
  }

  public Object get(String k) {
    return data.get(k);
  }

  public <T> T getType(String key) {
    return (T) data.get(key);
  }

  public <T> T getType(String key, Logger LOG) {
    try {
      return getType(key);
    } catch (ClassCastException e) {
      LOG.error("Failed to get " + key + " from message (Invalid type). " , e);
      return null;
    }
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("Message{");
    sb.append("data=").append(data);
    sb.append(", op=").append(op);
    sb.append('}');
    return sb.toString();
  }

  public String toJson() {
    return gson.toJson(this);
  }

  public static Message fromJson(String json) {
    return gson.fromJson(json, Message.class);
  }
}
