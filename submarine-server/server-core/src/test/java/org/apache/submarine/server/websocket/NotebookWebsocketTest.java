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
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.StatusCode;
import org.eclipse.jetty.websocket.api.WebSocketAdapter;
import org.eclipse.jetty.websocket.client.WebSocketClient;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.net.URI;
import java.util.concurrent.Future;


public class NotebookWebsocketTest {

  @BeforeClass
  public static void init() throws Exception {
    AbstractSubmarineServerTest.startUp(
        NotebookWebsocketTest.class.getSimpleName());
  }

  @AfterClass
  public static void destroy() throws Exception {
    AbstractSubmarineServerTest.shutDown();
  }

  @Test
  public void testWebsocketConnection() throws Exception{
    URI uri = URI.create(
        AbstractSubmarineServerTest.getWebsocketApiUrlToTest("notebook"));
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
      System.out.println("Socket Connected: " + sess);
    }

    @Override
    public void onWebSocketText(String message)
    {
      super.onWebSocketText(message);
      System.out.println("Received TEXT message: " + message);
    }

    @Override
    public void onWebSocketClose(int statusCode, String reason)
    {
      super.onWebSocketClose(statusCode, reason);
      System.out.println("Socket Closed: [" + statusCode + "] " + reason);
    }

    @Override
    public void onWebSocketError(Throwable cause)
    {
      super.onWebSocketError(cause);
      cause.printStackTrace(System.err);
    }
  }
}
