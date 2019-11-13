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

package org.apache.submarine.jobserver.rest.provider;

import org.yaml.snakeyaml.Yaml;

import javax.ws.rs.Consumes;
import javax.ws.rs.Produces;
import javax.ws.rs.WebApplicationException;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.MultivaluedMap;
import javax.ws.rs.ext.MessageBodyReader;
import javax.ws.rs.ext.MessageBodyWriter;
import javax.ws.rs.ext.Provider;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.lang.annotation.Annotation;
import java.lang.reflect.Type;
import java.util.Scanner;

@Provider
@Consumes({"application/yaml", MediaType.TEXT_PLAIN})
@Produces({"application/yaml", MediaType.TEXT_PLAIN})
public class YamlEntityProvider<T> implements MessageBodyWriter<T>, MessageBodyReader<T> {

  @Override
  public boolean isReadable(Class<?> type, Type genericType,
      Annotation[] annotations,
      MediaType mediaType) {
    return true;
  }

  @Override
  public T readFrom(Class<T> type, Type genericType, Annotation[] annotations,
      MediaType mediaType,
      MultivaluedMap<String, String> httpHeaders, InputStream entityStream)
      throws WebApplicationException {
    Yaml yaml = new Yaml();
    T t = yaml.loadAs(toString(entityStream), type);
    return t;
  }

  public static String toString(InputStream inputStream) {
    return new Scanner(inputStream, "UTF-8")
        .useDelimiter("\\A").next();
  }

  @Override
  public boolean isWriteable(Class<?> type, Type genericType,
      Annotation[] annotations,
      MediaType mediaType) {
    return true;
  }

  @Override
  public long getSize(T t, Class<?> type, Type genericType,
      Annotation[] annotations,
      MediaType mediaType) {
    return -1;
  }

  @Override
  public void writeTo(T t, Class<?> type, Type genericType,
      Annotation[] annotations,
      MediaType mediaType, MultivaluedMap<String, Object> httpHeaders,
      OutputStream entityStream) throws IOException, WebApplicationException {
    Yaml yaml = new Yaml();
    OutputStreamWriter writer = new OutputStreamWriter(entityStream);
    yaml.dump(t, writer);
    writer.close();
  }
}
