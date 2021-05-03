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
	clientset "submarine-cloud-v2/pkg/generated/clientset/versioned"
	submarinescheme "submarine-cloud-v2/pkg/generated/clientset/versioned/scheme"
	informers "submarine-cloud-v2/pkg/generated/informers/externalversions/submarine/v1alpha1"
	listers "submarine-cloud-v2/pkg/generated/listers/submarine/v1alpha1"
	v1alpha1 "submarine-cloud-v2/pkg/submarine/v1alpha1"
	"time"

	"submarine-cloud-v2/pkg/helm"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

	traefik "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/clientset/versioned"
	traefikinformers "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/informers/externalversions/traefik/v1alpha1"
	traefiklisters "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/listers/traefik/v1alpha1"
	traefikv1alpha1 "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/traefik/v1alpha1"
)

const controllerAgentName = "submarine-controller"

// Controller is the controller implementation for Foo resources
type Controller struct {
	// kubeclientset is a standard kubernetes clientset
	kubeclientset kubernetes.Interface
	// sampleclientset is a clientset for our own API group
	submarineclientset clientset.Interface
	traefikclientset   traefik.Interface

	submarinesLister listers.SubmarineLister
	submarinesSynced cache.InformerSynced

	deploymentLister            appslisters.DeploymentLister
	serviceaccountLister        corelisters.ServiceAccountLister
	serviceLister               corelisters.ServiceLister
	persistentvolumeLister      corelisters.PersistentVolumeLister
	persistentvolumeclaimLister corelisters.PersistentVolumeClaimLister
	ingressLister               extlisters.IngressLister
	ingressrouteLister          traefiklisters.IngressRouteLister
	clusterroleLister           rbaclisters.ClusterRoleLister
	clusterrolebindingLister    rbaclisters.ClusterRoleBindingLister
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
	traefikclientset traefik.Interface,
	deploymentInformer appsinformers.DeploymentInformer,
	serviceInformer coreinformers.ServiceInformer,
	serviceaccountInformer coreinformers.ServiceAccountInformer,
	persistentvolumeInformer coreinformers.PersistentVolumeInformer,
	persistentvolumeclaimInformer coreinformers.PersistentVolumeClaimInformer,
	ingressInformer extinformers.IngressInformer,
	ingressrouteInformer traefikinformers.IngressRouteInformer,
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
		kubeclientset:               kubeclientset,
		submarineclientset:          submarineclientset,
		traefikclientset:            traefikclientset,
		submarinesLister:            submarineInformer.Lister(),
		submarinesSynced:            submarineInformer.Informer().HasSynced,
		deploymentLister:            deploymentInformer.Lister(),
		serviceLister:               serviceInformer.Lister(),
		serviceaccountLister:        serviceaccountInformer.Lister(),
		persistentvolumeLister:      persistentvolumeInformer.Lister(),
		persistentvolumeclaimLister: persistentvolumeclaimInformer.Lister(),
		ingressLister:               ingressInformer.Lister(),
		ingressrouteLister:          ingressrouteInformer.Lister(),
		clusterroleLister:           clusterroleInformer.Lister(),
		clusterrolebindingLister:    clusterrolebindingInformer.Lister(),
		workqueue:                   workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "Submarines"),
		recorder:                    recorder,
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
func (c *Controller) newSubmarineServer(namespace string, serverImage string, serverReplicas int32) error {
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

// newSubmarineDatabase is a function to create submarine-database.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-database.yaml
func (c *Controller) newSubmarineDatabase(namespace string, spec *v1alpha1.SubmarineSpec) error {
	klog.Info("[newSubmarineDatabase]")
	databaseName := "submarine-database"

	databaseImage := spec.Database.Image
	if databaseImage == "" {
		databaseImage = "apache/submarine:database-" + spec.Version
	}

	// Step1: Create PersistentVolume
	// PersistentVolumes are not namespaced resources, so we add the namespace
	// as a suffix to distinguish them
	pvName := databaseName + "-pv--" + namespace
	pv, pv_err := c.persistentvolumeLister.Get(pvName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pv_err) {
		var persistentVolumeSource corev1.PersistentVolumeSource
		switch spec.Storage.StorageType {
		case "nfs":
			persistentVolumeSource = corev1.PersistentVolumeSource{
				NFS: &corev1.NFSVolumeSource{
					Server: spec.Storage.NfsIP,
					Path:   spec.Storage.NfsPath,
				},
			}
		case "host":
			hostPathType := corev1.HostPathDirectoryOrCreate
			persistentVolumeSource = corev1.PersistentVolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: spec.Storage.HostPath,
					Type: &hostPathType,
				},
			}
		default:
			klog.Warningln("	Invalid storageType found in submarine spec, nothing will be created!")
			return nil
		}
		pv, pv_err = c.kubeclientset.CoreV1().PersistentVolumes().Create(context.TODO(),
			&corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: pvName,
				},
				Spec: corev1.PersistentVolumeSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{
						corev1.ReadWriteMany,
					},
					Capacity: corev1.ResourceList{
						corev1.ResourceStorage: resource.MustParse(spec.Database.StorageSize),
					},
					PersistentVolumeSource: persistentVolumeSource,
				},
			},
			metav1.CreateOptions{})
		if pv_err != nil {
			klog.Info(pv_err)
		}
		klog.Info("	Create PersistentVolume: ", pv.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pv_err != nil {
		return pv_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	// Step2: Create PersistentVolumeClaim
	pvcName := databaseName + "-pvc"
	pvc, pvc_err := c.persistentvolumeclaimLister.PersistentVolumeClaims(namespace).Get(pvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pvc_err) {
		storageClassName := ""
		pvc, pvc_err = c.kubeclientset.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(),
			&corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: pvcName,
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{
						corev1.ReadWriteMany,
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceStorage: resource.MustParse(spec.Database.StorageSize),
						},
					},
					VolumeName:       pvName,
					StorageClassName: &storageClassName,
				},
			},
			metav1.CreateOptions{})
		if pvc_err != nil {
			klog.Info(pvc_err)
		}
		klog.Info("	Create PersistentVolumeClaim: ", pvc.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pvc_err != nil {
		return pvc_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	// Step3: Create Deployment
	deployment, deployment_err := c.deploymentLister.Deployments(namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(deployment_err) {
		deployment, deployment_err = c.kubeclientset.AppsV1().Deployments(namespace).Create(context.TODO(),
			&appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: databaseName,
				},
				Spec: appsv1.DeploymentSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": databaseName,
						},
					},
					Replicas: spec.Database.Replicas,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": databaseName,
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            databaseName,
									Image:           databaseImage,
									ImagePullPolicy: "IfNotPresent",
									Ports: []corev1.ContainerPort{
										{
											ContainerPort: 3306,
										},
									},
									Env: []corev1.EnvVar{
										{
											Name:  "MYSQL_ROOT_PASSWORD",
											Value: "password",
										},
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											MountPath: "/var/lib/mysql",
											Name:      "volume",
											SubPath:   databaseName,
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: "volume",
									VolumeSource: corev1.VolumeSource{
										PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
											ClaimName: pvcName,
										},
									},
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		if deployment_err != nil {
			klog.Info(deployment_err)
		}
		klog.Info("	Create Deployment: ", deployment.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if deployment_err != nil {
		return deployment_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293

	// Step4: Create Service
	service, service_err := c.serviceLister.Services(namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(service_err) {
		service, service_err = c.kubeclientset.CoreV1().Services(namespace).Create(context.TODO(),
			&corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: databaseName,
				},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       3306,
							TargetPort: intstr.FromInt(3306),
							Name:       databaseName,
						},
					},
					Selector: map[string]string{
						"app": databaseName,
					},
				},
			},
			metav1.CreateOptions{})
		if service_err != nil {
			klog.Info(service_err)
		}
		klog.Info("	Create Service: ", service.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if service_err != nil {
		return service_err
	}

	// TODO: (sample-controller) controller.go:287 ~ 293
	return nil
}

// subcharts: https://github.com/apache/submarine/tree/master/helm-charts/submarine/charts

func (c *Controller) newSubCharts(namespace string) error {
	// Install traefik
	// Reference: https://github.com/apache/submarine/tree/master/helm-charts/submarine/charts/traefik

	if !helm.CheckRelease("traefik", namespace) {
		klog.Info("[Helm] Install Traefik")
		helm.HelmInstallLocalChart(
			"traefik",
			"charts/traefik",
			"traefik",
			namespace,
			map[string]string{},
		)
	}

	if !helm.CheckRelease("notebook-controller", namespace) {
		klog.Info("[Helm] Install Notebook-Controller")
		helm.HelmInstallLocalChart(
			"notebook-controller",
			"charts/notebook-controller",
			"notebook-controller",
			namespace,
			map[string]string{},
		)
	}

	if !helm.CheckRelease("tfjob", namespace) {
		klog.Info("[Helm] Install TFjob")
		helm.HelmInstallLocalChart(
			"tfjob",
			"charts/tfjob",
			"tfjob",
			namespace,
			map[string]string{},
		)
	}

	// TODO: maintain "error"
	// TODO: (sample-controller) controller.go:287 ~ 293
	// TODO: port-forward
	// 		kubectl port-forward --address 0.0.0.0 service/traefik 32080:80

	return nil
}

// newSubmarineTensorboard is a function to create submarine-tensorboard.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-tensorboard.yaml
func (c *Controller) newSubmarineTensorboard(namespace string, spec *v1alpha1.SubmarineSpec) error {
	klog.Info("[newSubmarineTensorboard]")
	tensorboardName := "submarine-tensorboard"

	// Step 1: Create PersistentVolume
	// PersistentVolumes are not namespaced resources, so we add the namespace
	// as a suffix to distinguish them
	pvName := tensorboardName + "-pv--" + namespace
	pv, pv_err := c.persistentvolumeLister.Get(pvName)

	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pv_err) {
		var persistentVolumeSource corev1.PersistentVolumeSource
		switch spec.Storage.StorageType {
		case "nfs":
			persistentVolumeSource = corev1.PersistentVolumeSource{
				NFS: &corev1.NFSVolumeSource{
					Server: spec.Storage.NfsIP,
					Path:   spec.Storage.NfsPath,
				},
			}
		case "host":
			hostPathType := corev1.HostPathDirectoryOrCreate
			persistentVolumeSource = corev1.PersistentVolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: spec.Storage.HostPath,
					Type: &hostPathType,
				},
			}
		default:
			klog.Warningln("	Invalid storageType found in submarine spec, nothing will be created!")
			return nil
		}
		pv, pv_err = c.kubeclientset.CoreV1().PersistentVolumes().Create(context.TODO(),
			&corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: pvName,
				},
				Spec: corev1.PersistentVolumeSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{
						corev1.ReadWriteMany,
					},
					Capacity: corev1.ResourceList{
						corev1.ResourceStorage: resource.MustParse(spec.Tensorboard.StorageSize),
					},
					PersistentVolumeSource: persistentVolumeSource,
				},
			},
			metav1.CreateOptions{})
		if pv_err != nil {
			klog.Info(pv_err)
		}
		klog.Info("	Create PersistentVolume: ", pv.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pv_err != nil {
		return pv_err
	}

	// Step 2: Create PersistentVolumeClaim
	pvcName := tensorboardName + "-pvc"
	pvc, pvc_err := c.persistentvolumeclaimLister.PersistentVolumeClaims(namespace).Get(pvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pvc_err) {
		storageClassName := ""
		pvc, pvc_err = c.kubeclientset.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(),
			&corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: pvcName,
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{
						corev1.ReadWriteMany,
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceStorage: resource.MustParse(spec.Tensorboard.StorageSize),
						},
					},
					VolumeName:       pvName,
					StorageClassName: &storageClassName,
				},
			},
			metav1.CreateOptions{})
		if pvc_err != nil {
			klog.Info(pvc_err)
		}
		klog.Info("	Create PersistentVolumeClaim: ", pvc.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pvc_err != nil {
		return pvc_err
	}

	// Step 3: Create Deployment
	deployment, deployment_err := c.deploymentLister.Deployments(namespace).Get(tensorboardName)
	if errors.IsNotFound(deployment_err) {
		deployment, deployment_err = c.kubeclientset.AppsV1().Deployments(namespace).Create(context.TODO(),
			&appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: tensorboardName,
				},
				Spec: appsv1.DeploymentSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": tensorboardName + "-pod",
						},
					},
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": tensorboardName + "-pod",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  tensorboardName + "-container",
									Image: "tensorflow/tensorflow:1.11.0",
									Command: []string{
										"tensorboard",
										"--logdir=/logs",
										"--path_prefix=/tensorboard",
									},
									ImagePullPolicy: "IfNotPresent",
									Ports: []corev1.ContainerPort{
										{
											ContainerPort: 6006,
										},
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											MountPath: "/logs",
											Name:      "volume",
											SubPath:   tensorboardName,
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: "volume",
									VolumeSource: corev1.VolumeSource{
										PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
											ClaimName: pvcName,
										},
									},
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		if deployment_err != nil {
			klog.Info(deployment_err)
		}
		klog.Info("	Create Deployment: ", deployment.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if deployment_err != nil {
		return deployment_err
	}

	// Step 4: Create Service
	serviceName := tensorboardName + "-service"
	service, service_err := c.serviceLister.Services(namespace).Get(serviceName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(service_err) {
		service, service_err = c.kubeclientset.CoreV1().Services(namespace).Create(context.TODO(),
			&corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: serviceName,
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"app": tensorboardName + "-pod",
					},
					Ports: []corev1.ServicePort{
						{
							Protocol:   "TCP",
							Port:       8080,
							TargetPort: intstr.FromInt(6006),
						},
					},
				},
			},
			metav1.CreateOptions{})
		if service_err != nil {
			klog.Info(service_err)
		}
		klog.Info(" Create Service: ", service.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if service_err != nil {
		return service_err
	}

	// Step 5: Create IngressRoute
	ingressroute, ingressroute_err := c.ingressrouteLister.IngressRoutes(namespace).Get(tensorboardName + "-ingressroute")
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(ingressroute_err) {
		ingressroute, ingressroute_err = c.traefikclientset.TraefikV1alpha1().IngressRoutes(namespace).Create(context.TODO(),
			&traefikv1alpha1.IngressRoute{
				ObjectMeta: metav1.ObjectMeta{
					Name: tensorboardName + "-ingressroute",
				},
				Spec: traefikv1alpha1.IngressRouteSpec{
					EntryPoints: []string{
						"web",
					},
					Routes: []traefikv1alpha1.Route{
						{
							Kind:  "Rule",
							Match: "PathPrefix(`/tensorboard`)",
							Services: []traefikv1alpha1.Service{
								{
									LoadBalancerSpec: traefikv1alpha1.LoadBalancerSpec{
										Kind: "Service",
										Name: serviceName,
										Port: 8080,
									},
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		if ingressroute_err != nil {
			klog.Info(ingressroute_err)
		}
		klog.Info(" Create IngressRoute: ", ingressroute.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if ingressroute_err != nil {
		return ingressroute_err
	}

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

	// Install subcharts
	c.newSubCharts(namespace)

	// Create submarine-server
	serverImage := submarine.Spec.Server.Image
	serverReplicas := *submarine.Spec.Server.Replicas
	if serverImage == "" {
		serverImage = "apache/submarine:server-" + submarine.Spec.Version
	}
	err = c.newSubmarineServer(namespace, serverImage, serverReplicas)
	if err != nil {
		return err
	}

	// Create Submarine Database
	err = c.newSubmarineDatabase(namespace, &submarine.Spec)
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

	// Create Submarine Tensorboard
	err = c.newSubmarineTensorboard(namespace, &submarine.Spec)
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
