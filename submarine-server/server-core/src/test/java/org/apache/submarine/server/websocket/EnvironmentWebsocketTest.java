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

import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.SubmarineServer;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.StatusCode;
import org.eclipse.jetty.websocket.api.WebSocketAdapter;
import org.eclipse.jetty.websocket.client.WebSocketClient;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.util.concurrent.Future;

import static junit.framework.TestCase.assertEquals;


public class EnvironmentWebsocketTest {

  private static final Logger LOG = LoggerFactory.getLogger(EnvironmentWebsocketTest.class);
  @BeforeClass
  public static void init() throws Exception {
    AbstractSubmarineServerTest.startUp(
        EnvironmentWebsocketTest.class.getSimpleName());
  }

  @AfterClass
  public static void destroy() throws Exception {
    AbstractSubmarineServerTest.shutDown();
  }

  @Test
  public void testWebsocketConnection() throws Exception{
    URI uri = URI.create(
        AbstractSubmarineServerTest.getWebsocketApiUrlToTest("environment"));
    WebSocketClient client = new WebSocketClient();

    try {
      client.start();
      // The socket that receives events
      EventSocket socket = new EventSocket();
      // Attempt Connect
      Future<Session> fut = client.connect(socket, uri);
      // Wait for Connect
      Session session = fut.get();
      // Send a message
      session.getRemote().sendString("Hello");
      // Close session
      //session.close();
      session.close(StatusCode.NORMAL, "I'm done");
    } finally {
      client.stop();
    }
  }

  public class EventSocket extends WebSocketAdapter
  {
    @Override
    public void onWebSocketConnect(Session sess)
    {
      super.onWebSocketConnect(sess);
      LOG.info("Socket Connected: " + sess);
    }

    @Override
    public void onWebSocketText(String message)
    {
      super.onWebSocketText(message);
      LOG.info("Received TEXT message: " + message);
      assertEquals(message, "Hello");
    }

    @Override
    public void onWebSocketClose(int statusCode, String reason)
    {
      super.onWebSocketClose(statusCode, reason);
      LOG.info("Socket Closed: [" + statusCode + "] " + reason);
      assertEquals(statusCode, StatusCode.NORMAL);
      assertEquals(reason, "I'm done");
    }

    @Override
    public void onWebSocketError(Throwable cause)
    {
      super.onWebSocketError(cause);
      cause.printStackTrace(System.err);
    }
  }
}
