package controller

import (
	traefikv1alpha1 "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/traefik/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/client-go/tools/cache"
)

func (cb *ControllerBuilder) RegisterSubmarineEventHandlers() *ControllerBuilder {
	cb.actions["submarine"] = func() {
		cb.config.submarineInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: cb.controller.enqueueSubmarine,
			UpdateFunc: func(old, new interface{}) {
				cb.controller.enqueueSubmarine(new)
			},
		})
	}
	return cb
}

func (cb *ControllerBuilder) RegisterNamespaceEventHandlers() *ControllerBuilder {
	cb.actions["namespace"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterDeploymentEventHandlers() *ControllerBuilder {
	cb.actions["deployment"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterServiceEventHandlers() *ControllerBuilder {
	cb.actions["service"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterServiceAccountEventHandlers() *ControllerBuilder {
	cb.actions["serviceaccount"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterPersistentVolumeClaimEventHandlers() *ControllerBuilder {
	cb.actions["persistentvolumeclaim"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterIngressEventHandlers() *ControllerBuilder {
	cb.actions["ingress"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterIngressRouteEventHandlers() *ControllerBuilder {
	cb.actions["ingressroute"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterRoleEventHandlers() *ControllerBuilder {
	cb.actions["role"] = func() {
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
	}
	return cb
}

func (cb *ControllerBuilder) RegisterRoleBindingEventHandlers() *ControllerBuilder {
	cb.actions["rolebinding"] = func() {
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
	}
	return cb
}
