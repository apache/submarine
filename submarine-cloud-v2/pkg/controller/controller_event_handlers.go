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
	// traefikv1alpha1 "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/traefik/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/client-go/tools/cache"
)

func (cb *ControllerBuilder) addSubmarineEventHandlers() *ControllerBuilder {
	cb.config.submarineInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.enqueueSubmarine,
		UpdateFunc: func(old, new interface{}) {
			cb.controller.enqueueSubmarine(new)
		},
	})

	return cb
}

func (cb *ControllerBuilder) addNamespaceEventHandlers() *ControllerBuilder {
	cb.config.namespaceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newNamespace := new.(*corev1.Namespace)
			oldNamespace := old.(*corev1.Namespace)
			if newNamespace.ResourceVersion == oldNamespace.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addDeploymentEventHandlers() *ControllerBuilder {
	cb.config.deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newDeployment := new.(*appsv1.Deployment)
			oldDeployment := old.(*appsv1.Deployment)
			if newDeployment.ResourceVersion == oldDeployment.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addServiceEventHandlers() *ControllerBuilder {
	cb.config.serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newService := new.(*corev1.Service)
			oldService := old.(*corev1.Service)
			if newService.ResourceVersion == oldService.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addServiceAccountEventHandlers() *ControllerBuilder {
	cb.config.serviceaccountInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newServiceAccount := new.(*corev1.ServiceAccount)
			oldServiceAccount := old.(*corev1.ServiceAccount)
			if newServiceAccount.ResourceVersion == oldServiceAccount.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addPersistentVolumeClaimEventHandlers() *ControllerBuilder {
	cb.config.persistentvolumeclaimInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newPVC := new.(*corev1.PersistentVolumeClaim)
			oldPVC := old.(*corev1.PersistentVolumeClaim)
			if newPVC.ResourceVersion == oldPVC.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addConfigMapEventHandlers() *ControllerBuilder {
	cb.config.configMapInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newConfigMap := new.(*corev1.ConfigMap)
			oldConfigMap := old.(*corev1.ConfigMap)
			if newConfigMap.ResourceVersion == oldConfigMap.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addIngressEventHandlers() *ControllerBuilder {
	cb.config.ingressInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newIngress := new.(*extensionsv1beta1.Ingress)
			oldIngress := old.(*extensionsv1beta1.Ingress)
			if newIngress.ResourceVersion == oldIngress.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

/*
func (cb *ControllerBuilder) addIngressRouteEventHandlers() *ControllerBuilder {
	cb.config.ingressrouteInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newIngressRoute := new.(*traefikv1alpha1.IngressRoute)
			oldIngressRoute := old.(*traefikv1alpha1.IngressRoute)
			if newIngressRoute.ResourceVersion == oldIngressRoute.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}
*/

func (cb *ControllerBuilder) addRoleEventHandlers() *ControllerBuilder {
	cb.config.roleInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newRole := new.(*rbacv1.Role)
			oldRole := old.(*rbacv1.Role)
			if newRole.ResourceVersion == oldRole.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}

func (cb *ControllerBuilder) addRoleBindingEventHandlers() *ControllerBuilder {
	cb.config.rolebindingInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cb.controller.handleObject,
		UpdateFunc: func(old, new interface{}) {
			newRoleBinding := new.(*rbacv1.RoleBinding)
			oldRoleBinding := old.(*rbacv1.RoleBinding)
			if newRoleBinding.ResourceVersion == oldRoleBinding.ResourceVersion {
				return
			}
			cb.controller.handleObject(new)
		},
		DeleteFunc: cb.controller.handleObject,
	})

	return cb
}
