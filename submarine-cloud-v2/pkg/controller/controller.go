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
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1alpha1 "github.com/apache/submarine/submarine-cloud-v2/pkg/apis/submarine/v1alpha1"
	clientset "github.com/apache/submarine/submarine-cloud-v2/pkg/client/clientset/versioned"
	listers "github.com/apache/submarine/submarine-cloud-v2/pkg/client/listers/submarine/v1alpha1"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	extlisters "k8s.io/client-go/listers/extensions/v1beta1"
	rbaclisters "k8s.io/client-go/listers/rbac/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	traefik "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/clientset/versioned"
	traefiklisters "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/listers/traefik/v1alpha1"
)

const controllerAgentName = "submarine-controller"

const storageClassName = "submarine-storageclass"

const (
	serverName                  = "submarine-server"
	databaseName                = "submarine-database"
	databasePort                = 3306
	tensorboardName             = "submarine-tensorboard"
	mlflowName                  = "submarine-mlflow"
	minioName                   = "submarine-minio"
	ingressName                 = serverName + "-ingress"
	databasePvcName             = databaseName + "-pvc"
	tensorboardPvcName          = tensorboardName + "-pvc"
	tensorboardServiceName      = tensorboardName + "-service"
	tensorboardIngressRouteName = tensorboardName + "-ingressroute"
	mlflowPvcName               = mlflowName + "-pvc"
	mlflowServiceName           = mlflowName + "-service"
	mlflowIngressRouteName      = mlflowName + "-ingressroute"
	minioPvcName                = minioName + "-pvc"
	minioServiceName            = minioName + "-service"
	minioIngressRouteName       = minioName + "-ingressroute"
	artifactPath                = "./artifacts/submarine/"
	databaseYamlPath            = artifactPath + "submarine-database.yaml"
	ingressYamlPath             = artifactPath + "submarine-ingress.yaml"
	minioYamlPath               = artifactPath + "submarine-minio.yaml"
	mlflowYamlPath              = artifactPath + "submarine-mlflow.yaml"
	serverYamlPath              = artifactPath + "submarine-server.yaml"
	tensorboardYamlPath         = artifactPath + "submarine-tensorboard.yaml"
	rbacYamlPath                = artifactPath + "submarine-rbac.yaml"
)

var dependents = []string{serverName, databaseName, tensorboardName, mlflowName, minioName}

const (
	// SuccessSynced is used as part of the Event 'reason' when a Submarine is synced
	SuccessSynced = "Synced"
	// ErrResourceExists is used as part of the Event 'reason' when a Submarine fails
	// to sync due to a Deployment of the same name already existing.
	ErrResourceExists = "ErrResourceExists"

	// MessageResourceExists is the message used for Events when a resource
	// fails to sync due to a Deployment already existing
	MessageResourceExists = "Resource %q already exists and is not managed by Submarine"
	// MessageResourceSynced is the message used for an Event fired when a
	// Submarine is synced successfully
	MessageResourceSynced = "Submarine synced successfully"
)

// Controller is the controller implementation for Submarine resources
type Controller struct {
	// kubeclientset is a standard kubernetes clientset
	kubeclientset kubernetes.Interface
	// sampleclientset is a clientset for our own API group
	submarineclientset clientset.Interface
	traefikclientset   traefik.Interface

	submarinesLister listers.SubmarineLister
	submarinesSynced cache.InformerSynced

	namespaceLister             corelisters.NamespaceLister
	deploymentLister            appslisters.DeploymentLister
	serviceaccountLister        corelisters.ServiceAccountLister
	serviceLister               corelisters.ServiceLister
	persistentvolumeclaimLister corelisters.PersistentVolumeClaimLister
	ingressLister               extlisters.IngressLister
	ingressrouteLister          traefiklisters.IngressRouteLister
	roleLister                  rbaclisters.RoleLister
	rolebindingLister           rbaclisters.RoleBindingLister
	// workqueue is a rate limited work queue. This is used to queue work to be
	// processed instead of performing it as soon as a change happens. This
	// means we can ensure we only process a fixed amount of resources at a
	// time, and makes it easy to ensure we are never processing the same item
	// simultaneously in two different workers.
	workqueue workqueue.RateLimitingInterface
	// recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	recorder record.EventRecorder

	incluster bool
}

func (c *Controller) Run(threadiness int, stopCh <-chan struct{}) error {
	defer utilruntime.HandleCrash()
	defer c.workqueue.ShutDown()

	// Start the informer factories to begin populating the informer caches
	klog.Info("Starting Submarine controller")

	// Wait for the caches to be synced before starting workers
	klog.Info("Waiting for informer caches to sync")
	if ok := cache.WaitForCacheSync(stopCh, c.submarinesSynced); !ok {
		return fmt.Errorf("failed to wait for caches to sync")
	}

	klog.Info("Starting workers")
	// Launch $threadiness workers to process Submarine resources
	for i := 0; i < threadiness; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	klog.Info("Started workers")
	<-stopCh
	klog.Info("Shutting down workers")

	return nil
}

// runWorker is a long-running function that will continually call the
// processNextWorkItem function in order to read and process a message on the
// workqueue.
func (c *Controller) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem will read a single work item off the workqueue and
// attempt to process it, by calling the syncHandler.
func (c *Controller) processNextWorkItem() bool {
	obj, shutdown := c.workqueue.Get()
	if shutdown {
		return false
	}

	// We wrap this block in a func so we can defer c.workqueue.Done.
	err := func(obj interface{}) error {
		defer c.workqueue.Done(obj)
		var key string
		var ok bool
		if key, ok = obj.(string); !ok {
			// As the item in the workqueue is actually invalid, we call
			// Forget here else we'd go into a loop of attempting to
			// process a work item that is invalid.
			c.workqueue.Forget(obj)
			utilruntime.HandleError(fmt.Errorf("expected WorkQueueItem in workqueue but got %#v", obj))
			return nil
		}
		// Run the syncHandler
		if err := c.syncHandler(key); err != nil {
			// Put the item back on the workqueue to handle any transient errors.
			c.workqueue.AddRateLimited(key)
			return fmt.Errorf("error syncing '%s': %s, requeuing", key, err.Error())
		}
		// Finally, if no error occurs we Forget this item so it does not
		// get queued again until another change happens.
		c.workqueue.Forget(obj)
		klog.Infof("Successfully synced '%s'", key)
		return nil
	}(obj)

	if err != nil {
		utilruntime.HandleError(err)
		return true
	}

	return true
}

// syncHandler compares the actual state with the desired, and attempts to
// converge the two. It then updates the Status block of the Submarine resource
// with the current status of the resource.
// State Machine for Submarine
//+-----------------------------------------------------------------+
//|      +---------+         +----------+          +----------+     |
//|      |         |         |          |          |          |     |
//|      |   New   +---------> Creating +----------> Running  |     |
//|      |         |         |          |          |          |     |
//|      +----+----+         +-----+----+          +-----+----+     |
//|           |                    |                     |          |
//|           |                    |                     |          |
//|           |                    |                     |          |
//|           |                    |               +-----v----+     |
//|           |                    |               |          |     |
//|           +--------------------+--------------->  Failed  |     |
//|                                                |          |     |
//|                                                +----------+     |
//+-----------------------------------------------------------------+
func (c *Controller) syncHandler(key string) error {
	// Convert the namespace/name string into a distinct namespace and name
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("invalid resource key: %s", key))
		return nil
	}
	klog.Info("syncHandler: ", key)

	// Get the Submarine resource with this namespace/name
	submarine, err := c.getSubmarine(namespace, name)
	if err != nil {
		return err
	}
	if submarine == nil {
		// The Submarine resource may no longer exist, in which case we stop
		// processing
		utilruntime.HandleError(fmt.Errorf("submarine '%s' in work queue no longer exists", key))
		return nil
	}

	// Submarine is in the terminating process, only used when in foreground cascading deletion, otherwise the submarine will be recreated
	if !submarine.DeletionTimestamp.IsZero() {
		return nil
	}

	submarineCopy := submarine.DeepCopy()

	// Take action based on submarine state
	switch submarineCopy.Status.SubmarineState.State {
	case v1alpha1.NewState:
		c.recordSubmarineEvent(submarineCopy)
		if err := c.validateSubmarine(submarineCopy); err != nil {
			submarineCopy.Status.SubmarineState.State = v1alpha1.FailedState
			submarineCopy.Status.SubmarineState.ErrorMessage = err.Error()
			c.recordSubmarineEvent(submarineCopy)
		} else {
			submarineCopy.Status.SubmarineState.State = v1alpha1.CreatingState
			c.recordSubmarineEvent(submarineCopy)
		}
	case v1alpha1.CreatingState:
		if err := c.createSubmarine(submarineCopy); err != nil {
			submarineCopy.Status.SubmarineState.State = v1alpha1.FailedState
			submarineCopy.Status.SubmarineState.ErrorMessage = err.Error()
			c.recordSubmarineEvent(submarineCopy)
		}
		ok, err := c.checkSubmarineDependentsReady(submarineCopy)
		if err != nil {
			submarineCopy.Status.SubmarineState.State = v1alpha1.FailedState
			submarineCopy.Status.SubmarineState.ErrorMessage = err.Error()
			c.recordSubmarineEvent(submarineCopy)
		}
		if ok {
			submarineCopy.Status.SubmarineState.State = v1alpha1.RunningState
			c.recordSubmarineEvent(submarineCopy)
		}
	case v1alpha1.RunningState:
		if err := c.createSubmarine(submarineCopy); err != nil {
			submarineCopy.Status.SubmarineState.State = v1alpha1.FailedState
			submarineCopy.Status.SubmarineState.ErrorMessage = err.Error()
			c.recordSubmarineEvent(submarineCopy)
		}
	}

	// update submarine status
	err = c.updateSubmarineStatus(submarine, submarineCopy)
	if err != nil {
		return err
	}

	return nil
}

func (c *Controller) updateSubmarineStatus(submarine, submarineCopy *v1alpha1.Submarine) error {
	// Update server replicas
	serverDeployment, err := c.getDeployment(submarine.Namespace, serverName)
	if err != nil {
		return err
	}
	if serverDeployment != nil {
		submarineCopy.Status.AvailableServerReplicas = serverDeployment.Status.AvailableReplicas
	}

	// Update database replicas
	databaseDeployment, err := c.getDeployment(submarine.Namespace, databaseName)
	if err != nil {
		return err
	}
	if databaseDeployment != nil {
		submarineCopy.Status.AvailableDatabaseReplicas = databaseDeployment.Status.AvailableReplicas
	}

	// Skip update if nothing changed.
	if equality.Semantic.DeepEqual(submarine.Status, submarineCopy.Status) {
		return nil
	}

	_, err = c.submarineclientset.SubmarineV1alpha1().Submarines(submarine.Namespace).Update(context.TODO(), submarineCopy, metav1.UpdateOptions{})
	if err != nil {
		return err
	}
	return nil
}

// enqueueSubmarine takes a Submarine resource and converts it into a namespace/name
// string which is then put onto the work queue. This method should *not* be
// passed resources of any type other than Submarine.
func (c *Controller) enqueueSubmarine(obj interface{}) {
	var key string
	var err error
	if key, err = cache.MetaNamespaceKeyFunc(obj); err != nil {
		utilruntime.HandleError(err)
		return
	}

	// key: [namespace]/[CR name]
	// Example: default/example-submarine
	c.workqueue.Add(key)
}

// handleObject will take any resource implementing metav1.Object and attempt
// to find the Submarine resource that 'owns' it. It does this by looking at the
// objects metadata.ownerReferences field for an appropriate OwnerReference.
// It then enqueues that Submarine resource to be processed. If the object does not
// have an appropriate OwnerReference, it will simply be skipped.
func (c *Controller) handleObject(obj interface{}) {
	var object metav1.Object
	var ok bool
	if object, ok = obj.(metav1.Object); !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("error decoding object, invalid type"))
			return
		}
		object, ok = tombstone.Obj.(metav1.Object)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("error decoding object tombstone, invalid type"))
			return
		}
		klog.V(4).Infof("Recovered deleted object '%s' from tombstone", object.GetName())
	}
	klog.V(4).Infof("Processing object: %s", object.GetName())
	if ownerRef := metav1.GetControllerOf(object); ownerRef != nil {
		// If this object is not owned by a Submarine, we should not do anything
		// more with it.
		if ownerRef.Kind != "Submarine" {
			return
		}

		submarine, err := c.submarinesLister.Submarines(object.GetNamespace()).Get(ownerRef.Name)
		if err != nil {
			klog.V(4).Infof("ignoring orphaned object '%s' of submarine '%s'", object.GetSelfLink(), ownerRef.Name)
			return
		}

		c.enqueueSubmarine(submarine)
		return
	}
}

func (c *Controller) getSubmarine(namespace, name string) (*v1alpha1.Submarine, error) {
	submarine, err := c.submarinesLister.Submarines(namespace).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return submarine, nil
}

func (c *Controller) getDeployment(namespace, name string) (*appsv1.Deployment, error) {
	deployment, err := c.deploymentLister.Deployments(namespace).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return deployment, nil
}

func (c *Controller) validateSubmarine(submarine *v1alpha1.Submarine) error {

	// Print out the spec of the Submarine resource
	b, err := json.MarshalIndent(submarine.Spec, "", "  ")
	fmt.Println(string(b))

	if err != nil {
		return err
	}

	return nil
}

func (c *Controller) createSubmarine(submarine *v1alpha1.Submarine) error {
	var err error
	err = c.createSubmarineServer(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = c.createSubmarineDatabase(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = c.createIngress(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = c.createSubmarineServerRBAC(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = c.createSubmarineTensorboard(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = c.createSubmarineMlflow(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = c.createSubmarineMinio(submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	return nil
}

func (c *Controller) checkSubmarineDependentsReady(submarine *v1alpha1.Submarine) (bool, error) {
	for _, name := range dependents {
		deployment, err := c.getDeployment(submarine.Namespace, name)
		if err != nil {
			return false, err
		}
		// check if deployment replicas failed
		for _, condition := range deployment.Status.Conditions {
			if condition.Type == appsv1.DeploymentReplicaFailure {
				return false, fmt.Errorf("failed creating replicas of %s, message: %s", deployment.Name, condition.Message)
			}
		}
		// check if ready replicas are same as targeted replicas
		if deployment.Status.ReadyReplicas != deployment.Status.Replicas {
			return false, nil
		}
	}

	return true, nil
}

func (c *Controller) recordSubmarineEvent(submarine *v1alpha1.Submarine) {
	switch submarine.Status.SubmarineState.State {
	case v1alpha1.NewState:
		c.recorder.Eventf(
			submarine,
			corev1.EventTypeNormal,
			"SubmarineAdded",
			"Submarine %s was added",
			submarine.Name)
	case v1alpha1.CreatingState:
		c.recorder.Eventf(
			submarine,
			corev1.EventTypeNormal,
			"SubmarineCreating",
			"Submarine %s was creating",
			submarine.Name,
		)
	case v1alpha1.RunningState:
		c.recorder.Eventf(
			submarine,
			corev1.EventTypeNormal,
			"SubmarineRunning",
			"Submarine %s was running",
			submarine.Name,
		)
	case v1alpha1.FailedState:
		c.recorder.Eventf(
			submarine,
			corev1.EventTypeWarning,
			"SubmarineFailed",
			"Submarine %s was failed: %s",
			submarine.Name,
			submarine.Status.SubmarineState.ErrorMessage,
		)
	}
}
