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
	"fmt"
	"time"

	"github.com/golang/glog"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	typedcorev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"

	stablev1 "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	clientset "github.com/apache/submarine/submarine-cloud/pkg/client/clientset/versioned"
	studentscheme "github.com/apache/submarine/submarine-cloud/pkg/client/clientset/versioned/scheme"
	informers "github.com/apache/submarine/submarine-cloud/pkg/client/informers/externalversions/submarine/v1alpha1"
	listers "github.com/apache/submarine/submarine-cloud/pkg/client/listers/submarine/v1alpha1"
)

const controllerName = "submarine-controller"

const (
	SuccessSynced         = "Synced"
	MessageResourceSynced = "SubmarineServer Object synced successfully"
	MessageTest           = "Someone changed the email in a distributed system," + controllerName + "automatically completed the evolution to the end state"
)

// Controller is the controller implementation for Student resources
type SubmarineController struct {
	// k8s's clientset
	kubeclientset kubernetes.Interface
	// our own API group的clientset
	submarineServerClientset clientset.Interface

	submarineServerLister listers.SubmarineServerLister
	submarineServerSynced cache.InformerSynced

	workQueue workqueue.RateLimitingInterface

	syncHandler func(dKey string) error

	recorder record.EventRecorder
}

// New SubmarineController
func NewSubmarineController(
	kubeclientset kubernetes.Interface,
	studentclientset clientset.Interface,
	studentInformer informers.SubmarineServerInformer) *SubmarineController {

	utilruntime.Must(studentscheme.AddToScheme(scheme.Scheme))
	glog.Info("Create event broadcaster")
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&typedcorev1.EventSinkImpl{Interface: kubeclientset.CoreV1().Events("")})

	// FIXME Create a source is the controllerAgentName event logger, which is itself part of the audit / log
	// kubectl describe crd test1
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: controllerName})
	controller := &SubmarineController{
		kubeclientset:            kubeclientset,
		submarineServerClientset: studentclientset,
		submarineServerLister:    studentInformer.Lister(),
		submarineServerSynced:    studentInformer.Informer().HasSynced,
		workQueue:                workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "Students"),
		recorder:                 recorder,
	}

	glog.Info("Listen for SubmarineServer's add / update / delete events")

	studentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controller.addSubmarineServerHandler,
		UpdateFunc: func(old, new interface{}) {
			oldSubmarineServer := old.(*stablev1.SubmarineServer)
			newSubmarineServer := new.(*stablev1.SubmarineServer)
			// The versions are the same, which means that there is no actual update operation
			if oldSubmarineServer.ResourceVersion == newSubmarineServer.ResourceVersion {
				return
			}
			controller.addSubmarineServerHandler(new)
		},
		DeleteFunc: controller.deleteSubmarineServerHandler,
	})

	controller.syncHandler = controller.syncSubmarineServer

	return controller
}

// Start your controller business here
func (c *SubmarineController) Run(max int, stopCh <-chan struct{}) error {
	defer runtime.HandleCrash()
	defer c.workQueue.ShutDown()

	glog.Info("Start the controller event and start a cache data synchronization")
	if ok := cache.WaitForCacheSync(stopCh, c.submarineServerSynced); !ok {
		return fmt.Errorf("failed to wait for caches to sync")
	}

	glog.Infof("Start %d worker", max)
	for i := 0; i < max; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	glog.Info("worker all started")

	<-stopCh

	glog.Info("worker all stop")
	glog.Infof("%s final.", controllerName)

	return nil
}

func (c *SubmarineController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// worker Fetching data
func (c *SubmarineController) processNextWorkItem() bool {
	// Get a student from the queue
	obj, shutdown := c.workQueue.Get()
	if shutdown {
		return false
	}

	err := func(obj interface{}) error {
		defer c.workQueue.Done(obj)
		var key string
		var ok bool

		if key, ok = obj.(string); !ok {
			c.workQueue.Forget(obj)
			runtime.HandleError(fmt.Errorf("expected string in workQueue but got %#v", obj))
			return nil
		}
		// Handling event in syncHandler
		if err := c.syncSubmarineServer(key); err != nil {
			return fmt.Errorf("error syncing '%s': %s", key, err.Error())
		}

		c.workQueue.Forget(obj)
		glog.Infof("Successfully synced '%s'", key)
		return nil
	}(obj)

	if err != nil {
		runtime.HandleError(err)
		return true
	}

	return true
}

// Specific process
func (c *SubmarineController) syncSubmarineServer(key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		runtime.HandleError(fmt.Errorf("invalid resource key: %s", key))
		return nil
	}

	// Fetching objects from the cache
	submarineServer, err := c.submarineServerLister.SubmarineServers(namespace).Get(name)
	if err != nil {
		// If the Student object is deleted, it will come here, so you should add execution here
		if errors.IsNotFound(err) {
			glog.Infof("SubmarineServer object is deleted, please perform the actual deletion event here: %s/%s ...", namespace, name)

			return nil
		}

		runtime.HandleError(fmt.Errorf("failed to list SubmarineServer by: %s/%s", namespace, name))

		return err
	}

	glog.Infof("Here is the actual state of the SubmarineServer object: %v ...", submarineServer)
	submarineServerCopy := submarineServer.DeepCopy()

	// FIXME Here is the simulated end-state business
	if submarineServerCopy.Spec.Name == "gfandada" && submarineServerCopy.Spec.Email != "gfandada@gmail.com" {
		glog.Infof("===========================================================================================================")
		submarineServerCopy.Spec.Email = "gfandada@gmail.com"
		glog.Infof("Expected state%v", submarineServerCopy)
		glog.Infof("name=%v resourceVersion=%v ns=%v 类型=%v owner=%v uid=%v", submarineServerCopy.Name, submarineServerCopy.ResourceVersion,
			submarineServerCopy.Namespace, submarineServerCopy.Kind, submarineServerCopy.OwnerReferences, submarineServerCopy.UID)

		// FIXME  crd's curd generally needs to be encapsulated, you can refer to the encapsulation of Deployment
		result := &stablev1.SubmarineServer{}
		c.submarineServerClientset.SubmarineV1alpha1().RESTClient().Put().
			Namespace(submarineServerCopy.Namespace).
			Resource("submarineservers").Name(submarineServerCopy.Name).Body(submarineServerCopy).Do().Into(result)
		c.recorder.Event(submarineServer, corev1.EventTypeWarning, SuccessSynced, MessageTest)
		glog.Infof("===========================================================================================================")
		return nil
	}
	c.recorder.Event(submarineServer, corev1.EventTypeNormal, SuccessSynced, MessageResourceSynced)
	return nil
}

// Add SubmarineServer
func (c *SubmarineController) addSubmarineServerHandler(obj interface{}) {
	var key string
	var err error

	// Put objects in cache
	if key, err = cache.MetaNamespaceKeyFunc(obj); err != nil {
		runtime.HandleError(err)
		return
	}
	// Put the key into the queue
	c.workQueue.AddRateLimited(key)
}

// Delete SubmarineServer
func (c *SubmarineController) deleteSubmarineServerHandler(obj interface{}) {
	var key string
	var err error

	// Removes the specified object from the cache
	key, err = cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		runtime.HandleError(err)
		return
	}
	// Put the key into the queue
	c.workQueue.AddRateLimited(key)
}
