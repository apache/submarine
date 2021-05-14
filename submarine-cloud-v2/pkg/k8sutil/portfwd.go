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

// Reference: https://github.com/minio/operator/blob/master/kubectl-minio/cmd/proxy.go#L131

package k8sutil

import (
	"context"
	"fmt"
	"github.com/fatih/color"
	"io"
	"log"
	"os/exec"
	"strconv"
	"sync"
)

// run the command inside a goroutine, return a channel that closes then the command dies
func ServicePortForwardPort(ctx context.Context, namespace string, service string, localPort int, remotePort int, dcolor color.Attribute) chan interface{} {
	ch := make(chan interface{})
	go func() {
		defer close(ch)
		// service we are going to forward
		serviceName := fmt.Sprintf("service/%s", service)
		// command to run
		portStr := strconv.Itoa(localPort) + ":" + strconv.Itoa(remotePort)
		cmd := exec.CommandContext(ctx, "kubectl", "port-forward", "--address", "0.0.0.0", "-n", namespace, serviceName, portStr)
		// prepare to capture the output
		var errStdout, errStderr error
		stdoutIn, _ := cmd.StdoutPipe()
		stderrIn, _ := cmd.StderrPipe()
		err := cmd.Start()
		if err != nil {
			log.Fatalf("cmd.Start() failed with '%s'\n", err)
		}

		// cmd.Wait() should be called only after we finish reading
		// from stdoutIn and stderrIn.
		// wg ensures that we finish
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			errStdout = copyAndCapture(stdoutIn, dcolor)
			wg.Done()
		}()

		errStderr = copyAndCapture(stderrIn, dcolor)

		wg.Wait()

		err = cmd.Wait()
		if err != nil {
			log.Printf("cmd.Run() failed with %s\n", err.Error())
			return
		}
		if errStdout != nil || errStderr != nil {
			log.Printf("failed to capture stdout or stderr\n")
			return
		}
	}()
	return ch
}

// capture and print the output of the command
func copyAndCapture(r io.Reader, dcolor color.Attribute) error {
	var out []byte
	buf := make([]byte, 1024)
	for {
		n, err := r.Read(buf[:])
		if n > 0 {
			d := buf[:n]
			out = append(out, d...)
			theColor := color.New(dcolor)
			_, err := theColor.Print(string(d))

			if err != nil {
				return err
			}
		}
		if err != nil {
			// Read returns io.EOF at the end of file, which is not an error for us
			if err == io.EOF {
				err = nil
			}
			return err
		}
	}
}
