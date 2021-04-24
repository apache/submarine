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
	"encoding/json"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "submarine-cloud-v2/pkg/generated/clientset/versioned"
	submarinescheme "submarine-cloud-v2/pkg/generated/clientset/versioned/scheme"
	informers "submarine-cloud-v2/pkg/generated/informers/externalversions/submarine/v1alpha1"
	listers "submarine-cloud-v2/pkg/generated/listers/submarine/v1alpha1"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	extinformers "k8s.io/client-go/informers/extensions/v1beta1"
	rbacinformers "k8s.io/client-go/informers/rbac/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	typedcorev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	extlisters "k8s.io/client-go/listers/extensions/v1beta1"
	rbaclisters "k8s.io/client-go/listers/rbac/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const controllerAgentName = "submarine-controller"

// Controller is the controller implementation for Foo resources
type Controller struct {
	// kubeclientset is a standard kubernetes clientset
	kubeclientset kubernetes.Interface
	// sampleclientset is a clientset for our own API group
	submarineclientset clientset.Interface

	submarinesLister listers.SubmarineLister
	submarinesSynced cache.InformerSynced

	deploymentLister         appslisters.DeploymentLister
	serviceaccountLister     corelisters.ServiceAccountLister
	serviceLister            corelisters.ServiceLister
	ingressLister            extlisters.IngressLister
	clusterroleLister        rbaclisters.ClusterRoleLister
	clusterrolebindingLister rbaclisters.ClusterRoleBindingLister
	// workqueue is a rate limited work queue. This is used to queue work to be
	// processed instead of performing it as soon as a change happens. This
	// means we can ensure we only process a fixed amount of resources at a
	// time, and makes it easy to ensure we are never processing the same item
	// simultaneously in two different workers.
	workqueue workqueue.RateLimitingInterface
	// recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	recorder record.EventRecorder
}

// NewController returns a new sample controller
func NewController(
	kubeclientset kubernetes.Interface,
	submarineclientset clientset.Interface,
	deploymentInformer appsinformers.DeploymentInformer,
	serviceInformer coreinformers.ServiceInformer,
	serviceaccountInformer coreinformers.ServiceAccountInformer,
	ingressInformer extinformers.IngressInformer,
	clusterroleInformer rbacinformers.ClusterRoleInformer,
	clusterrolebindingInformer rbacinformers.ClusterRoleBindingInformer,
	submarineInformer informers.SubmarineInformer) *Controller {

	// TODO: Create event broadcaster
	// Add Submarine types to the default Kubernetes Scheme so Events can be
	// logged for Submarine types.
	utilruntime.Must(submarinescheme.AddToScheme(scheme.Scheme))
	klog.V(4).Info("Creating event broadcaster")
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartStructuredLogging(0)
	eventBroadcaster.StartRecordingToSink(&typedcorev1.EventSinkImpl{Interface: kubeclientset.CoreV1().Events("")})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: controllerAgentName})

	// Initialize controller
	controller := &Controller{
		kubeclientset:            kubeclientset,
		submarineclientset:       submarineclientset,
		submarinesLister:         submarineInformer.Lister(),
		submarinesSynced:         submarineInformer.Informer().HasSynced,
		deploymentLister:         deploymentInformer.Lister(),
		serviceLister:            serviceInformer.Lister(),
		serviceaccountLister:     serviceaccountInformer.Lister(),
		ingressLister:            ingressInformer.Lister(),
		clusterroleLister:        clusterroleInformer.Lister(),
		clusterrolebindingLister: clusterrolebindingInformer.Lister(),
		workqueue:                workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "Submarines"),
		recorder:                 recorder,
	}

	// Setting up event handler for Submarine
	klog.Info("Setting up event handlers")
	submarineInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controller.enqueueSubmarine,
		UpdateFunc: func(old, new interface{}) {
			controller.enqueueSubmarine(new)
		},
	})

	// TODO: Setting up event handler for other resources. E.g. namespace

	return controller
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
	// Launch two workers to process Submarine resources
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
		// TODO: Maintain workqueue
		defer c.workqueue.Done(obj)
		key, _ := obj.(string)
		c.syncHandler(key)
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

// newSubmarineServer is a function to create submarine-server.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-server.yaml
func (c *Controller) newSubmarineServer(serverImage string, serverReplicas int32, namespace string) error {
	klog.Info("[newSubmarineServer]")
	serverName := "submarine-server"

	// Step1: Create ServiceAccount
	serviceaccount, serviceaccount_err := c.serviceaccountLister.ServiceAccounts(namespace).Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(serviceaccount_err) {
		serviceaccount, serviceaccount_err = c.kubeclientset.CoreV1().ServiceAccounts(namespace).Create(context.TODO(),
			&corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name: serverName,
				},
			},
			metav1.CreateOptions{})
		klog.Info("	Create ServiceAccount: ", serviceaccount.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if serviceaccount_err != nil {
		return serviceaccount_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	// Step2: Create Service
	service, service_err := c.serviceLister.Services(namespace).Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(service_err) {
		service, service_err = c.kubeclientset.CoreV1().Services(namespace).Create(context.TODO(),
			&corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: serverName,
					Labels: map[string]string{
						"run": serverName,
					},
				},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       8080,
							TargetPort: intstr.FromInt(8080),
							Protocol:   "TCP",
						},
					},
					Selector: map[string]string{
						"run": serverName,
					},
				},
			},
			metav1.CreateOptions{})
		klog.Info("	Create Service: ", service.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if service_err != nil {
		return service_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	// Step3: Create Deployment
	deployment, deployment_err := c.deploymentLister.Deployments(namespace).Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(deployment_err) {
		deployment, deployment_err = c.kubeclientset.AppsV1().Deployments(namespace).Create(context.TODO(),
			&appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: serverName,
				},
				Spec: appsv1.DeploymentSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"run": serverName,
						},
					},
					Replicas: &serverReplicas,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"run": serverName,
							},
						},
						Spec: corev1.PodSpec{
							ServiceAccountName: serverName,
							Containers: []corev1.Container{
								{
									Name:  serverName,
									Image: serverImage,
									Env: []corev1.EnvVar{
										{
											Name:  "SUBMARINE_SERVER_PORT",
											Value: "8080",
										},
										{
											Name:  "SUBMARINE_SERVER_PORT_8080_TCP",
											Value: "8080",
										},
										{
											Name:  "SUBMARINE_SERVER_DNS_NAME",
											Value: serverName + "." + namespace,
										},
										{
											Name:  "K8S_APISERVER_URL",
											Value: "kubernetes.default.svc",
										},
									},
									Ports: []corev1.ContainerPort{
										{
											ContainerPort: 8080,
										},
									},
									ImagePullPolicy: "IfNotPresent",
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		klog.Info("	Create Deployment: ", deployment.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if deployment_err != nil {
		return deployment_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	return nil
}

// newIngress is a function to create Ingress.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-ingress.yaml
func (c *Controller) newIngress(namespace string) error {
	klog.Info("[newIngress]")
	serverName := "submarine-server"

	// Step1: Create ServiceAccount
	ingress, ingress_err := c.ingressLister.Ingresses(namespace).Get(serverName + "-ingress")
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(ingress_err) {
		ingress, ingress_err = c.kubeclientset.ExtensionsV1beta1().Ingresses(namespace).Create(context.TODO(),
			&extensionsv1beta1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      serverName + "-ingress",
					Namespace: namespace,
				},
				Spec: extensionsv1beta1.IngressSpec{
					Rules: []extensionsv1beta1.IngressRule{
						{
							IngressRuleValue: extensionsv1beta1.IngressRuleValue{
								HTTP: &extensionsv1beta1.HTTPIngressRuleValue{
									Paths: []extensionsv1beta1.HTTPIngressPath{
										{
											Backend: extensionsv1beta1.IngressBackend{
												ServiceName: serverName,
												ServicePort: intstr.FromInt(8080),
											},
											Path: "/",
										},
									},
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		klog.Info("	Create Ingress: ", ingress.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if ingress_err != nil {
		return ingress_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	return nil
}

// newSubmarineServerRBAC is a function to create RBAC for submarine-server.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/rbac.yaml
func (c *Controller) newSubmarineServerRBAC(serviceaccount_namespace string) error {
	klog.Info("[newSubmarineServerRBAC]")
	serverName := "submarine-server"
	// Step1: Create ClusterRole
	clusterrole, clusterrole_err := c.clusterroleLister.Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(clusterrole_err) {
		clusterrole, clusterrole_err = c.kubeclientset.RbacV1().ClusterRoles().Create(context.TODO(),
			&rbacv1.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: serverName,
				},
				Rules: []rbacv1.PolicyRule{
					{
						Verbs:     []string{"get", "list", "watch", "create", "delete", "deletecollection", "patch", "update"},
						APIGroups: []string{"kubeflow.org"},
						Resources: []string{"tfjobs", "tfjobs/status", "pytorchjobs", "pytorchjobs/status", "notebooks", "notebooks/status"},
					},
					{
						Verbs:     []string{"get", "list", "watch", "create", "delete", "deletecollection", "patch", "update"},
						APIGroups: []string{"traefik.containo.us"},
						Resources: []string{"ingressroutes"},
					},
					{
						Verbs:     []string{"*"},
						APIGroups: []string{""},
						Resources: []string{"pods", "pods/log", "services", "persistentvolumes", "persistentvolumeclaims"},
					},
					{
						Verbs:     []string{"*"},
						APIGroups: []string{"apps"},
						Resources: []string{"deployments", "deployments/status"},
					},
				},
			},
			metav1.CreateOptions{})
		klog.Info("	Create ClusterRole: ", clusterrole.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if clusterrole_err != nil {
		return clusterrole_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	clusterrolebinding, clusterrolebinding_err := c.clusterrolebindingLister.Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(clusterrolebinding_err) {
		clusterrolebinding, clusterrolebinding_err = c.kubeclientset.RbacV1().ClusterRoleBindings().Create(context.TODO(),
			&rbacv1.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: serverName,
				},
				Subjects: []rbacv1.Subject{
					rbacv1.Subject{
						Kind:      "ServiceAccount",
						Namespace: serviceaccount_namespace,
						Name:      serverName,
					},
				},
				RoleRef: rbacv1.RoleRef{
					Kind:     "ClusterRole",
					Name:     serverName,
					APIGroup: "rbac.authorization.k8s.io",
				},
			},
			metav1.CreateOptions{})
		klog.Info("	Create ClusterRoleBinding: ", clusterrolebinding.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if clusterrolebinding_err != nil {
		return clusterrolebinding_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	return nil
}

// syncHandler compares the actual state with the desired, and attempts to
// converge the two. It then updates the Status block of the Foo resource
// with the current status of the resource.
func (c *Controller) syncHandler(key string) error {
	// TODO: business logic

	// Convert the namespace/name string into a distinct namespace and name
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Invalid resource key: %s", key))
		return nil
	}

	// Get the Submarine resource with this namespace/name
	submarine, err := c.submarinesLister.Submarines(namespace).Get(name)
	if err != nil {
		// The Submarine resource may no longer exist, in which case we stop
		// processing
		if errors.IsNotFound(err) {
			utilruntime.HandleError(fmt.Errorf("submarine '%s' in work queue no longer exists", key))
			return nil
		}
	}

	klog.Info("syncHandler: ", key)

	// Print out the spec of the Submarine resource
	b, err := json.MarshalIndent(submarine.Spec, "", "  ")
	fmt.Println(string(b))

	// Create submarine-server
	serverImage := submarine.Spec.Server.Image
	serverReplicas := *submarine.Spec.Server.Replicas
	if serverImage == "" {
		serverImage = "apache/submarine:server-" + submarine.Spec.Version
	}

	// Create Submarine Server
	err = c.newSubmarineServer(serverImage, serverReplicas, namespace)
	if err != nil {
		return err
	}

	// Create ingress
	err = c.newIngress(namespace)
	if err != nil {
		return err
	}

	// Create RBAC
	err = c.newSubmarineServerRBAC(namespace)
	if err != nil {
		return err
	}

	return nil
}

// enqueueFoo takes a Submarine resource and converts it into a namespace/name
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
