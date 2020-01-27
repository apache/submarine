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
package submarine

import (
	"bytes"
	"encoding/json"
	"github.com/golang/glog"
	"io"
	"net/http"
	"time"
)

// ClientInterface submarine client interface
type ClientInterface interface {
	// Close closes the connection.
	Close() error

	// Cmd calls the given Submarine command.
	///Cmd(cmd string, args ...interface{}) *submarine.Resp

	// PipeAppend adds the given call to the pipeline queue.
	// Use PipeResp() to read the response.
	///PipeAppend(cmd string, args ...interface{})

	// PipeResp returns the reply for the next request in the pipeline queue. Err
	// with ErrPipelineEmpty is returned if the pipeline queue is empty.
	///PipeResp() *submarine.Resp

	// PipeClear clears the contents of the current pipeline queue, both commands
	// queued by PipeAppend which have yet to be sent and responses which have yet
	// to be retrieved through PipeResp. The first returned int will be the number
	// of pending commands dropped, the second will be the number of pending
	// responses dropped
	PipeClear() (int, int)

	// ReadResp will read a Resp off of the connection without sending anything
	// first (useful after you've sent a SUSBSCRIBE command). This will block until
	// a reply is received or the timeout is reached (returning the IOErr). You can
	// use IsTimeout to check if the Resp is due to a Timeout
	//
	// Note: this is a more low-level function, you really shouldn't have to
	// actually use it unless you're writing your own pub/sub code
	///ReadResp() *submarine.Resp

	// GetClusterAddress calls the given Submarine cluster server address list
	GetClusterAddress() ([]string, error)
}

// Client structure representing a client connection to submarine
type Client struct {
	commandsMapping map[string]string
	///client          *submarine.Client
	client ClientInterface
}

const getClusterAddressUrl = "/api/v1/cluster/address"
const getClusterNodesUrl = "/api/v1/cluster/nodes"

// NewClient build a client connection and connect to a submarine address
func NewClient(addr string, cnxTimeout time.Duration, commandsMapping map[string]string) (ClientInterface, error) {
	var err error
	c := &Client{
		commandsMapping: commandsMapping,
	}

	// c.client, err = submarine.DialTimeout("tcp", addr, cnxTimeout)
	// TODO error!!!!

	return c.client, err
}

// GetClusterAddress calls the given Submarine cluster server address list.
func (c *Client) GetClusterAddress(host string) ([]string, error) {
	clusterAddrBuff := httpGet(host + getClusterAddressUrl)
	var clusterAddress []string
	err := json.Unmarshal(clusterAddrBuff.Bytes(), &clusterAddress)
	if err != nil {
		glog.Error("Unmarshal failure: %s", clusterAddrBuff.String())
	}

	return clusterAddress, nil
}

func httpGet(url string) *bytes.Buffer {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	var buffer [4096]byte
	result := bytes.NewBuffer(nil)
	for {
		n, err := resp.Body.Read(buffer[0:])
		result.Write(buffer[0:n])
		if err != nil && err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
	}

	return result
}
