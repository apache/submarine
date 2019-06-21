/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.apache.submarine.tony.proxy;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;


/**
 * This class is used to proxy requests from gateway to the cluster hosts.
 * Initial purpose of this class to to proxy requests to gateway to the hosts.
 */
public class ProxyServer {

  private static final Log LOG = LogFactory.getLog(ProxyServer.class);
  private String remoteHost;
  private int remotePort;
  private int localPort;
  public ProxyServer(String remoteHost, int remotePort, int localPort) {
    this.remoteHost = remoteHost;
    this.remotePort = remotePort;
    this.localPort = localPort;
  }
  public void start() throws IOException {
    LOG.info("Starting proxy for " + remoteHost + ":" + remotePort
             + " on port " + localPort);
    ServerSocket server = new ServerSocket(localPort);
    while (true) {
      new Proxy(server.accept(), remoteHost, remotePort).start();
    }
  }

  static class Proxy extends Thread {
    private Socket clientSocket;
    private final String serverHost;
    private final int serverPort;
    Proxy(Socket client, String host, int port) {
      this.clientSocket = client;
      this.serverHost = host;
      this.serverPort = port;
    }

    @Override
    public void run() {
      try (Socket server = new Socket(serverHost, serverPort);
          final InputStream inFromClient = clientSocket.getInputStream();
          final OutputStream outToClient = clientSocket.getOutputStream();
          final InputStream inFromServer = server.getInputStream();
          final OutputStream outToServer = server.getOutputStream();
      ) {
        final byte[] request = new byte[1024];
        byte[] reply = new byte[4096];
        new Thread(() -> {
          int bytesRead;
          try {
            while ((bytesRead = inFromClient.read(request)) != -1) {
              outToServer.write(request, 0, bytesRead);
              outToServer.flush();
            }
            outToServer.close();
          } catch (IOException e) {
            LOG.error(e);
          }
        }).start();
        int bytesRead;
        while ((bytesRead = inFromServer.read(reply)) != -1) {
          outToClient.write(reply, 0, bytesRead);
          outToClient.flush();
        }
      } catch (IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      } finally {
        try {
          clientSocket.close();
        } catch (IOException e) {
          LOG.error(e);
        }
      }
    }

  }
}
