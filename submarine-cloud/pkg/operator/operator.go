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

package operator

import (
	"github.com/apache/submarine/submarine-cloud/pkg/client"
	"github.com/apache/submarine/submarine-cloud/pkg/controller"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"time"

	submarineInformers "github.com/apache/submarine/submarine-cloud/pkg/client/informers/externalversions"
	kubeinformers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"

	"github.com/golang/glog"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

// Controller is the controller implementation for Student resources
type SubmarineOperator struct {
	kubeInformerFactory      kubeinformers.SharedInformerFactory
	submarineInformerFactory submarineInformers.SharedInformerFactory
	controller               *controller.Controller
}

func NewSubmarineOperator(cfg *Config) *SubmarineOperator {
	kubeConfig, err := initKubeConfig(cfg)
	if err != nil {
		glog.Fatalf("Unable to init submarinecluster controller: %v", err)
	}

	extClient, err := apiextensionsclient.NewForConfig(kubeConfig)
	if err != nil {
		glog.Fatalf("Unable to init submarinClientset from kubeconfig:%v", err)
	}
	_, err = client.DefineSubmarineClusterResource(extClient)
	if err != nil && !apierrors.IsAlreadyExists(err) {
		glog.Fatalf("Unable to define SubmarineCluster resource:%v", err)
	}

	kubeClient, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		glog.Fatalf("Unable to initialize kubeClient:%v", err)
	}

	submarineClient, err := client.NewClient(kubeConfig)
	if err != nil {
		glog.Fatalf("Unable to init submarine.submarinClientset from kubeconfig:%v", err)
	}

	kubeInformerFactory := kubeinformers.NewSharedInformerFactory(kubeClient, time.Second*30)
	submarineInformerFactory := submarineInformers.NewSharedInformerFactory(submarineClient, time.Second*30)
	op := &SubmarineOperator{
		kubeInformerFactory:      kubeInformerFactory,
		submarineInformerFactory: submarineInformerFactory,
		controller:               controller.NewController(controller.NewConfig(1, cfg.Submarine), kubeClient, submarineClient, kubeInformerFactory, submarineInformerFactory),
	}

	return op
}

func initKubeConfig(c *Config) (*rest.Config, error) {
	if len(c.KubeConfigFile) > 0 {
		return clientcmd.BuildConfigFromFlags(c.Master, c.KubeConfigFile) // out of cluster config
	}
	return rest.InClusterConfig()
}

// Run executes the Submarine Operator
func (op *SubmarineOperator) Run(stop <-chan struct{}) error {
	var err error
	if op.controller != nil {
		op.kubeInformerFactory.Start(stop)
		op.submarineInformerFactory.Start(stop)
		err = op.controller.Run(stop)
	}

	return err
}
