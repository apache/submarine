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
	"submarine-cloud-v2/pkg/generated/clientset/versioned/typed/submarine/v1alpha1"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/kubernetes"
    "testing"
    "reflect"
    "os"
)


var (
    masterURL = ""
    kubeconfig = os.Getenv("HOME") + "/.kube/config"
    namespace = "default"
)

func TestSubmarineClient(t *testing.T) {
    ctx := context.Background()
	config, e := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if e != nil {
		panic(e.Error())
	}
	client, e := v1alpha1.NewForConfig(config)
    if e != nil {
		panic(e.Error())
	}
    
	submarineList, e := client.Submarines(namespace).List(ctx, metav1.ListOptions{})
    if e != nil {
		panic(e.Error())
	}
    fmt.Println(reflect.TypeOf(submarineList), e)    
    // fmt.Println(submarineList, e)
}

func TestK8sClient(t *testing.T) {
    ctx := context.Background()
	config, e := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if e != nil {
		panic(e.Error())
	}
	client, e := kubernetes.NewForConfig(config)
    if e != nil {
		panic(e.Error())
	}
    PodsList, e := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
    fmt.Println(reflect.TypeOf(PodsList), e)    
    // fmt.Println(PodsList, e)
}
