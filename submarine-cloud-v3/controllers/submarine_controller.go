/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
)

// SubmarineReconciler reconciles a Submarine object
type SubmarineReconciler struct {
	// Fields required by the operator
	client.Client
	Scheme   *runtime.Scheme
	Log      logr.Logger
	Recorder record.EventRecorder
	// Fields required by submarine
	ClusterType             string
	CreatePodSecurityPolicy bool
}

// kubebuilder RBAC markers generates rules for the operator ClusterRole
// On change, run `make manifest` to update config/rbac/role.yaml
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/config/rbac/role.yaml

// Submarine resource
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines/finalizers,verbs=update

// Event
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the Submarine object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.11.0/pkg/reconcile
func (r *SubmarineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// TODO(user): your logic here

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *SubmarineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&submarineapacheorgv1alpha1.Submarine{}).
		Complete(r)
}
