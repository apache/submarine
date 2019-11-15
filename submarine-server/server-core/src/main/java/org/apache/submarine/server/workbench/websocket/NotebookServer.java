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
package org.apache.submarine.server.workbench.websocket;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.util.Date;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.commons.lang.StringUtils;
import org.eclipse.jetty.util.annotation.ManagedAttribute;
import org.eclipse.jetty.util.annotation.ManagedObject;
import org.eclipse.jetty.util.annotation.ManagedOperation;
import org.eclipse.jetty.websocket.servlet.WebSocketServlet;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Submarine websocket service. This class used setter injection because all servlet should have
 * no-parameter constructor
 */
@ManagedObject
public class NotebookServer extends WebSocketServlet
    implements NotebookSocketListener {

  /**
   * Job manager service type.
   */
  protected enum JobManagerServiceType {
    JOB_MANAGER_PAGE("JOB_MANAGER_PAGE");
    private String serviceTypeKey;

    JobManagerServiceType(String serviceType) {
      this.serviceTypeKey = serviceType;
    }

    String getKey() {
      return this.serviceTypeKey;
    }
  }

  private static final Logger LOG = LoggerFactory.getLogger(NotebookServer.class);
  private static Gson gson = new GsonBuilder()
      .setDateFormat("yyyy-MM-dd'T'HH:mm:ssZ")
      .registerTypeAdapter(Date.class, new DateJsonDeserializer())
      .setPrettyPrinting()
      .create();

  private static AtomicReference<NotebookServer> self = new AtomicReference<>();

  private ConnectionManager connectionManager;

  private ExecutorService executorService = Executors.newFixedThreadPool(10);

  public NotebookServer() {
    this.connectionManager = new ConnectionManager();
    NotebookServer.self.set(this);
    LOG.info("NotebookServer instantiated: {}", this);
  }

  @Override
  public void configure(WebSocketServletFactory factory) {
    factory.setCreator(new NotebookWebSocketCreator(this));
  }

  @Override
  public void onOpen(NotebookSocket conn) {
    LOG.info("New connection from {}", conn);
    connectionManager.addConnection(conn);
  }

  @Override
  public void onMessage(NotebookSocket conn, String msg) {
    try {
      LOG.info("Got Message: " + msg);
      if (StringUtils.isEmpty(conn.getUser())) {
        connectionManager.addUserConnection("FakeUser1", conn);
      }
    } catch (Exception e) {
      LOG.error("Can't handle message: " + msg, e);
      try {
        conn.send(serializeMessage(new Message(Message.OP.ERROR_INFO).put(
            "info", e.getMessage())));
      } catch (IOException iox) {
        LOG.error("Fail to send error info", iox);
      }
    }
  }

  @Override
  public void onClose(NotebookSocket conn, int code, String reason) {
    LOG.info("Closed connection to {} ({}) {}", conn, code, reason);
    connectionManager.removeConnection(conn);
    connectionManager.removeUserConnection(conn.getUser(), conn);
  }

  public ConnectionManager getConnectionManager() {
    return connectionManager;
  }

  protected Message deserializeMessage(String msg) {
    return gson.fromJson(msg, Message.class);
  }

  protected String serializeMessage(Message m) {
    return gson.toJson(m);
  }

  public void broadcast(Message m) {
    connectionManager.broadcast(m);
  }

  @ManagedAttribute
  public Set<String> getConnectedUsers() {
    return connectionManager.getConnectedUsers();
  }

  @ManagedOperation
  public void sendMessage(String message) {
    Message m = new Message(Message.OP.NOTICE);
    m.data.put("notice", message);
    connectionManager.broadcast(m);
  }
}
