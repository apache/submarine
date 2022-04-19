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

import com.google.common.collect.Sets;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.eclipse.jetty.websocket.api.WebSocketException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Date;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Manager class for managing websocket connections.
 */
public class ConnectionManager {
  private static final Logger LOG = LoggerFactory.getLogger(ConnectionManager.class);
  private static final Gson gson = new GsonBuilder()
      .setDateFormat("yyyy-MM-dd'T'HH:mm:ssZ")
      .registerTypeAdapter(Date.class, new DateJsonDeserializer())
      .setPrettyPrinting()
      .create();

  final Queue<WebSocket> connectedSockets = new ConcurrentLinkedQueue<>();
  // user -> connection
  final Map<String, Queue<WebSocket>> userSocketMap = new ConcurrentHashMap<>();

  public void addConnection(WebSocket conn) {
    connectedSockets.add(conn);
  }

  public void removeConnection(WebSocket conn) {
    connectedSockets.remove(conn);
  }

  public void addUserConnection(String user, WebSocket conn) {
    LOG.info("Add user connection {} for user: {}", conn, user);
    conn.setUser(user);
    if (userSocketMap.containsKey(user)) {
      userSocketMap.get(user).add(conn);
    } else {
      Queue<WebSocket> socketQueue = new ConcurrentLinkedQueue<>();
      socketQueue.add(conn);
      userSocketMap.put(user, socketQueue);
    }
  }

  public void removeUserConnection(String user, WebSocket conn) {
    LOG.info("Remove user connection {} for user: {}", conn, user);
    if (userSocketMap.containsKey(user)) {
      userSocketMap.get(user).remove(conn);
    } else {
      LOG.warn("Closing connection that is absent in user connections");
    }
  }

  protected String serializeMessage(Message m) {
    return gson.toJson(m);
  }

  public void broadcast(Message m) {
    synchronized (connectedSockets) {
      for (WebSocket ns : connectedSockets) {
        try {
          ns.send(serializeMessage(m));
        } catch (IOException | WebSocketException e) {
          LOG.error("Send error: " + m, e);
        }
      }
    }
  }

  public Set<String> getConnectedUsers() {
    Set<String> connectedUsers = Sets.newHashSet();
    for (WebSocket notebookSocket : connectedSockets) {
      connectedUsers.add(notebookSocket.getUser());
    }
    return connectedUsers;
  }
}
