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
package org.apache.submarine.server.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.ProtocolException;
import java.net.URL;
import java.net.URLConnection;
import java.util.Date;
import java.util.Map;

public class GitHttpRequest {
  private static final Logger LOG = LoggerFactory.getLogger(GitHttpRequest.class);

  /**
   * Sends an HTTP request to the specified URL
   * @param url
   * @param parameter
   * @param method
   * @return
   * @throws Exception
   */
  public static String sendHttpRequest(String url, Map<String, String> headParams,
                                       byte[] parameter, String method) {
    // Create a URLConnection
    URLConnection urlConnection = null;
    try {
      urlConnection = (new URL(url)).openConnection();
    } catch (IOException e) {
      LOG.error(e.getMessage(), e);
    }
    // Defines StringBuilder to facilitate string concatenation later
    // when reading web pages and returning byte stream information
    StringBuilder stringBuilder = new StringBuilder();

    // Force URLConnection class to HttpURLConnection class
    HttpURLConnection httpURLConnection = (HttpURLConnection) urlConnection;
    httpURLConnection.setDoInput(true);
    httpURLConnection.setDoOutput(true);
    // Set the request, which could be delete put post get
    try {
      httpURLConnection.setRequestMethod(method);
    } catch (ProtocolException e) {
      LOG.error(e.getMessage(), e);
    }
    if (parameter != null) {
      // Sets the length of the content
      httpURLConnection.setRequestProperty("Content-Length", String.valueOf(parameter.length));
    }
    // Set encoding format
    httpURLConnection.setRequestProperty("Content-Type", "application/json;charset=utf-8");
    // Sets the format of the receive return parameter
    httpURLConnection.setRequestProperty("accept", "application/json");
    httpURLConnection.setRequestProperty("connection", "Keep-Alive");
    httpURLConnection.setRequestProperty("user-agent",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1;SV1)");
    // Timestamp (accurate to seconds)
    httpURLConnection.setRequestProperty("timestamp", String.valueOf(new Date().getTime() / 1000));

    if (headParams != null) {
      for (String key : headParams.keySet()) {
        httpURLConnection.setRequestProperty(key, headParams.get(key));
      }
    }
    httpURLConnection.setUseCaches(false);

    if (parameter != null) {
      try (BufferedOutputStream outputStream =
               new BufferedOutputStream(httpURLConnection.getOutputStream())) {
        outputStream.write(parameter);
        outputStream.flush();
      } catch (IOException e) {
        LOG.error(e.getMessage(), e);
      }
    }
    try (InputStreamReader inputStreamReader =
             new InputStreamReader(httpURLConnection.getInputStream(), "utf-8");
         BufferedReader bufferedReader = new BufferedReader(inputStreamReader)) {
      String line;
      while ((line = bufferedReader.readLine()) != null) {
        stringBuilder.append(line);
      }
    } catch (IOException e) {
      LOG.error(e.getMessage(), e);
    }

    LOG.info("result:{}", stringBuilder.toString());
    return stringBuilder.toString();
  }
}
