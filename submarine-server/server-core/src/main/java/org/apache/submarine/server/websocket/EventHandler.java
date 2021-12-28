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

package org.apache.submarine.server.websocket;

import java.io.IOException;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketClose;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketConnect;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketError;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketMessage;
import org.eclipse.jetty.websocket.api.annotations.WebSocket;

@WebSocket
public class EventHandler {

  @OnWebSocketClose
  public void onClose(int statusCode, String reason) {
    System.out.println("Close: statusCode=" + statusCode + ", reason=" + reason);
  }

  @OnWebSocketError
  public void onError(Throwable t) {
    System.out.println("Error: " + t.getMessage());
  }

  @OnWebSocketConnect
  public void onConnect(Session session) {
    System.out.println("Connect: " + session.getRemoteAddress().getAddress());
    try {
      session.getRemote().sendString("Hello Webbrowser");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @OnWebSocketMessage
  public void onMessage(String message) {
    System.out.println("Message: " + message);
  }
}
