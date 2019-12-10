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

package controller

import (
	"fmt"
	"github.com/apache/submarine/submarine-cloud/pkg/controller/pod"
	"github.com/golang/glog"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/scheme"
	typedcorev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	policyv1listers "k8s.io/client-go/listers/policy/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"time"

	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	sClient "github.com/apache/submarine/submarine-cloud/pkg/client/clientset/versioned"
	sInformers "github.com/apache/submarine/submarine-cloud/pkg/client/informers/externalversions"
	sListers "github.com/apache/submarine/submarine-cloud/pkg/client/listers/submarine/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
)

// Controller contains all controller fields
type Controller struct {
	kubeClient      clientset.Interface
	submarineClient sClient.Interface

	submarineClusterLister sListers.SubmarineClusterLister
	submarineClusterSynced cache.InformerSynced

	podLister corev1listers.PodLister
	PodSynced cache.InformerSynced

	serviceLister corev1listers.ServiceLister
	ServiceSynced cache.InformerSynced

	podDisruptionBudgetLister  policyv1listers.PodDisruptionBudgetLister
	PodDiscruptionBudgetSynced cache.InformerSynced

	podControl                 pod.SubmarineClusterControlInteface
	serviceControl             ServicesControlInterface
	podDisruptionBudgetControl PodDisruptionBudgetsControlInterface

	updateHandler func(cluster *rapi.SubmarineCluster) (*rapi.SubmarineCluster, error) // callback to update SubmarineCluster. Added as member for testing

	queue    workqueue.RateLimitingInterface // SubmarineClusters to be synced
	recorder record.EventRecorder

	config *Config
}

// NewController builds and return new controller instance
func NewController(cfg *Config, kubeClient clientset.Interface, submarineClient sClient.Interface, kubeInformer kubeinformers.SharedInformerFactory, rInformer sInformers.SharedInformerFactory) *Controller {
	glog.Info("NewController()")
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&typedcorev1.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	serviceInformer := kubeInformer.Core().V1().Services()
	podInformer := kubeInformer.Core().V1().Pods()
	submarineInformer := rInformer.Submarine().V1alpha1().SubmarineClusters()
	podDisruptionBudgetInformer := kubeInformer.Policy().V1beta1().PodDisruptionBudgets()

	ctrl := &Controller{
		kubeClient:                 kubeClient,
		submarineClient:            submarineClient,
		submarineClusterLister:     submarineInformer.Lister(),
		submarineClusterSynced:     submarineInformer.Informer().HasSynced,
		podLister:                  podInformer.Lister(),
		PodSynced:                  podInformer.Informer().HasSynced,
		serviceLister:              serviceInformer.Lister(),
		ServiceSynced:              serviceInformer.Informer().HasSynced,
		podDisruptionBudgetLister:  podDisruptionBudgetInformer.Lister(),
		PodDiscruptionBudgetSynced: podDisruptionBudgetInformer.Informer().HasSynced,

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "submarinecluster"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, apiv1.EventSource{Component: "submarinecluster-controller"}),

		config: cfg,
	}

	submarineInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    ctrl.onAddSubmarineCluster,
			UpdateFunc: ctrl.onUpdateSubmarineCluster,
			DeleteFunc: ctrl.onDeleteSubmarineCluster,
		},
	)

	podInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    ctrl.onAddPod,
			UpdateFunc: ctrl.onUpdatePod,
			DeleteFunc: ctrl.onDeletePod,
		},
	)

	ctrl.updateHandler = ctrl.updateSubmarineCluster
	ctrl.podControl = pod.NewSubmarineClusterControl(ctrl.podLister, ctrl.kubeClient, ctrl.recorder)
	ctrl.serviceControl = NewServicesControl(ctrl.kubeClient, ctrl.recorder)
	ctrl.podDisruptionBudgetControl = NewPodDisruptionBudgetsControl(ctrl.kubeClient, ctrl.recorder)

	return ctrl
}

// Run executes the Controller
func (c *Controller) Run(stop <-chan struct{}) error {
	glog.Infof("Starting SubmarineCluster controller")

	if !cache.WaitForCacheSync(stop, c.PodSynced, c.submarineClusterSynced, c.ServiceSynced) {
		return fmt.Errorf("Timed out waiting for caches to sync")
	}

	for i := 0; i < c.config.NbWorker; i++ {
		go wait.Until(c.runWorker, time.Second, stop)
	}

	<-stop
	return nil
}

func (c *Controller) runWorker() {
	for c.processNextItem() {
	}
}

func (c *Controller) processNextItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)
	needRequeue, err := c.sync(key.(string))
	if err == nil {
		c.queue.Forget(key)
	} else {
		utilruntime.HandleError(fmt.Errorf("Error syncing submarinecluster: %v", err))
		c.queue.AddRateLimited(key)
		return true
	}

	if needRequeue {
		glog.V(4).Info("processNextItem: Requeue key:", key)
		c.queue.AddRateLimited(key)
	}

	return true
}

func (c *Controller) sync(key string) (bool, error) {
	glog.V(2).Infof("sync() key:%s", key)
	startTime := metav1.Now()
	defer func() {
		glog.V(2).Infof("Finished syncing SubmarineCluster %q (%v", key, time.Since(startTime.Time))
	}()
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	glog.V(6).Infof("Syncing %s/%s", namespace, name)
	sharedSubmarineCluster, err := c.submarineClusterLister.SubmarineClusters(namespace).Get(name)
	if err != nil {
		glog.Errorf("unable to get SubmarineCluster %s/%s: %v. Maybe deleted", namespace, name, err)
		return false, nil
	}

	if !rapi.IsSubmarineClusterDefaulted(sharedSubmarineCluster) {
		defaultedSubmarineCluster := rapi.DefaultSubmarineCluster(sharedSubmarineCluster)
		if _, err = c.updateHandler(defaultedSubmarineCluster); err != nil {
			glog.Errorf("SubmarineCluster %s/%s updated error:, err", namespace, name)
			return false, fmt.Errorf("unable to default SubmarineCluster %s/%s: %v", namespace, name, err)
		}
		glog.V(6).Infof("SubmarineCluster-Operator.sync Defaulted %s/%s", namespace, name)
		return false, nil
	}

	// TODO add validation

	// TODO: add test the case of graceful deletion
	if sharedSubmarineCluster.DeletionTimestamp != nil {
		return false, nil
	}

	submarinecluster := sharedSubmarineCluster.DeepCopy()

	// Init status.StartTime
	if submarinecluster.Status.StartTime == nil {
		submarinecluster.Status.StartTime = &startTime
		if _, err := c.updateHandler(submarinecluster); err != nil {
			glog.Errorf("SubmarineCluster %s/%s: unable init startTime: %v", namespace, name, err)
			return false, nil
		}
		glog.V(4).Infof("SubmarineCluster %s/%s: startTime updated", namespace, name)
		return false, nil
	}
	return c.syncCluster(submarinecluster)
}

func (c *Controller) syncCluster(submarineCluster *rapi.SubmarineCluster) (forceRequeue bool, err error) {
	glog.Infof("syncCluster()")
	return false, nil
}

func (c *Controller) onAddSubmarineCluster(obj interface{}) {
	glog.Infof("onAddSubmarineCluster(%v)", obj)
}

func (c *Controller) onDeleteSubmarineCluster(obj interface{}) {
	glog.Infof("onDeleteSubmarineCluster(%v)", obj)
}

func (c *Controller) onUpdateSubmarineCluster(oldObj, newObj interface{}) {
	glog.Infof("onUpdateSubmarineCluster(%v, %v)", oldObj, newObj)
}

func (c *Controller) onAddPod(obj interface{}) {
	glog.Infof("onAddPod(%v)", obj)
}

func (c *Controller) onUpdatePod(oldObj, newObj interface{}) {
	glog.Infof("onUpdatePod()")
}

func (c *Controller) onDeletePod(obj interface{}) {
	glog.Infof("onDeletePod()")
}

func (c *Controller) updateSubmarineCluster(submarineCluster *rapi.SubmarineCluster) (*rapi.SubmarineCluster, error) {
	rc, err := c.submarineClient.SubmarineV1alpha1().SubmarineClusters(submarineCluster.Namespace).Update(submarineCluster)
	if err != nil {
		glog.Errorf("updateSubmarineCluster cluster: [%v] error: %v", *submarineCluster, err)
		return rc, err
	}

	glog.V(6).Infof("SubmarineCluster %s/%s updated", submarineCluster.Namespace, submarineCluster.Name)
	return rc, nil
}
