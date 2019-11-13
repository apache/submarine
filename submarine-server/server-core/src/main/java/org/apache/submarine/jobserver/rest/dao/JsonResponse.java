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

package org.apache.submarine.jobserver.rest.dao;

import com.google.common.annotations.VisibleForTesting;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.core.NewCookie;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Json response builder.
 *
 * @param <T> can be an object or a ListResult
 */
public class JsonResponse<T> {
  private static final Logger LOG = LoggerFactory.getLogger(JsonResponse.class);

  private final javax.ws.rs.core.Response.Status status;
  private final int code;
  private final Boolean success;
  private final String message;
  private final T result;
  private final transient ArrayList<NewCookie> cookies;
  private final transient boolean pretty = false;
  private final Map<String, Object> attributes;

  private static Gson safeGson = null;

  private static final String CGLIB_PROPERTY_PREFIX = "\\$cglib_prop_";

  private JsonResponse(Builder builder) {
    this.status = builder.status;
    this.code = builder.code;
    this.success = builder.success;
    this.message = builder.message;
    this.attributes = builder.attributes;
    this.result = (T) builder.result;
    this.cookies = builder.cookies;
  }

  public T getResult() {
    return result;
  }

  public Boolean getSuccess() {
    return success;
  }

  @VisibleForTesting
  public Map<String, Object> getAttributes() {
    return attributes;
  }

  public static class Builder<T> {
    private javax.ws.rs.core.Response.Status status;
    private int code;
    private Boolean success;
    private String message;
    private T result;
    private Map<String, Object> attributes = new HashMap<>();
    private transient ArrayList<NewCookie> cookies;
    private transient boolean pretty = false;

    public Builder(javax.ws.rs.core.Response.Status status) {
      this.status = status;
      this.code = status.getStatusCode();
    }

    public Builder(int code) {
      this.code = code;
    }

    public Builder attribute(String key, Object value) {
      this.attributes.put(key, value);
      return this;
    }

    public Builder success(Boolean success) {
      this.success = success;
      return this;
    }

    public Builder message(String message) {
      this.message = message;
      return this;
    }

    public Builder result(T result) {
      this.result = result;
      return this;
    }

    public Builder code(int code) {
      this.code = code;
      return this;
    }

    public Builder cookies(ArrayList<NewCookie> newCookies) {
      if (cookies == null) {
        cookies = new ArrayList<>();
      }
      cookies.addAll(newCookies);
      return this;
    }

    public javax.ws.rs.core.Response build() {
      JsonResponse jsonResponse = new JsonResponse(this);
      return jsonResponse.build();
    }
  }

  @Override
  public String toString() {
    if (safeGson == null) {
      GsonBuilder gsonBuilder = new GsonBuilder();
      if (pretty) {
        gsonBuilder.setPrettyPrinting();
      }
      gsonBuilder.setExclusionStrategies(new JsonExclusionStrategy());

      // Trick to get the DefaultDateTypeAdatpter instance
      // Create a first instance a Gson
      Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

      // Get the date adapter
      TypeAdapter<Date> dateTypeAdapter = gson.getAdapter(Date.class);

      // Ensure the DateTypeAdapter is null safe
      TypeAdapter<Date> safeDateTypeAdapter = dateTypeAdapter.nullSafe();

      safeGson = new GsonBuilder()
          .registerTypeAdapter(Date.class, safeDateTypeAdapter)
          .serializeNulls().create();
    }

//    boolean haveDictAnnotation = false;
//    try {
//      if (null != getResult()) {
//        haveDictAnnotation = DictAnnotation.parseDictAnnotation(getResult());
//      }
//    } catch (Exception e) {
//      LOG.error(e.getMessage(), e);
//    }

    String json = safeGson.toJson(this);
//    if (haveDictAnnotation) {
//      json = json.replaceAll(CGLIB_PROPERTY_PREFIX, "");
//    }

    return json;
  }

  private synchronized javax.ws.rs.core.Response build() {
    Response.ResponseBuilder r = javax.ws.rs.core.Response.status(status).entity(this.toString());
    if (cookies != null) {
      for (NewCookie nc : cookies) {
        r.cookie(nc);
      }
    }
    return r.build();
  }

  // list result type response
  // Used to return a list of records
  public static class ListResult<T> {
    private List<T> records;
    private long total;

    public ListResult(List<T> records, long total) {
      this.records = records;
      this.total = total;
    }

    public List<T> getRecords() {
      return records;
    }

    public void setRecords(List<T> records) {
      this.records = records;
    }

    public long getTotal() {
      return total;
    }

    public void setTotal(long total) {
      this.total = total;
    }
  }
}
