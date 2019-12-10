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

package main

import (
	"context"
	goflag "flag"
	"fmt"
	"github.com/apache/submarine/submarine-cloud/pkg/operator"
	"github.com/apache/submarine/submarine-cloud/pkg/signal"
	"github.com/apache/submarine/submarine-cloud/pkg/utils"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"os"
	"runtime"
)

func main() {
	utils.BuildInfos()
	runtime.GOMAXPROCS(runtime.NumCPU())

	config := operator.NewSubmarineOperatorConfig()
	config.AddFlags(pflag.CommandLine)

	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	pflag.Parse()
	goflag.CommandLine.Parse([]string{})

	fmt.Println("config:", config)

	op := operator.NewSubmarineOperator(config)

	if err := run(op); err != nil {
		glog.Errorf("SubmarineOperator returns an error:%v", err)
		os.Exit(1)
	}

	os.Exit(0)
}

func run(op *operator.SubmarineOperator) error {
	ctx, cancelFunc := context.WithCancel(context.Background())
	go signal.HandleSignal(cancelFunc)

	op.Run(ctx.Done())

	return nil
}
