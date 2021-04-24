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
	"flag"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
	"os"
	clientset "submarine-cloud-v2/pkg/generated/clientset/versioned"
	informers "submarine-cloud-v2/pkg/generated/informers/externalversions"
	"submarine-cloud-v2/pkg/signals"
	"time"
)

var (
	masterURL  string
	kubeconfig string
	incluster  bool
)

func initKubeConfig() (*rest.Config, error) {
	if !incluster {
		return clientcmd.BuildConfigFromFlags(masterURL, kubeconfig) // out-of-cluster config
	}
	return rest.InClusterConfig() // in-cluster config
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// set up signals so we handle the first shutdown signal gracefully
	stopCh := signals.SetupSignalHandler()

	cfg, err := initKubeConfig()

	if err != nil {
		klog.Fatalf("Error building kubeconfig: %s", err.Error())
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		klog.Fatalf("Error building kubernetes clientset: %s", err.Error())
	}

	submarineClient, err := clientset.NewForConfig(cfg)
	if err != nil {
		klog.Fatalf("Error building submarine clientset: %s", err.Error())
	}

	kubeInformerFactory := kubeinformers.NewSharedInformerFactory(kubeClient, time.Second*30)
	submarineInformerFactory := informers.NewSharedInformerFactory(submarineClient, time.Second*30)

	// TODO: Pass informers to NewController()
	//       ex: namespace informer

	// Create a Submarine operator
	controller := NewController(kubeClient, submarineClient,
		kubeInformerFactory.Apps().V1().Deployments(),
		kubeInformerFactory.Core().V1().Services(),
		kubeInformerFactory.Core().V1().ServiceAccounts(),
		kubeInformerFactory.Extensions().V1beta1().Ingresses(),
		kubeInformerFactory.Rbac().V1().ClusterRoles(),
		kubeInformerFactory.Rbac().V1().ClusterRoleBindings(),
		submarineInformerFactory.Submarine().V1alpha1().Submarines())

	// notice that there is no need to run Start methods in a separate goroutine. (i.e. go kubeInformerFactory.Start(stopCh)
	// Start method is non-blocking and runs all registered informers in a dedicated goroutine.
	kubeInformerFactory.Start(stopCh)
	submarineInformerFactory.Start(stopCh)

	// Run controller
	if err = controller.Run(2, stopCh); err != nil {
		klog.Fatalf("Error running controller: %s", err.Error())
	}
}

func init() {
	flag.BoolVar(&incluster, "incluster", false, "Run submarine-operator in-cluster")
	flag.StringVar(&kubeconfig, "kubeconfig", os.Getenv("HOME")+"/.kube/config", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
}
