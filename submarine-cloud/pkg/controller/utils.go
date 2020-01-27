package controller

import (
	"errors"
	"fmt"
	"github.com/golang/glog"
	apiv1 "k8s.io/api/core/v1"
	"net"
	"time"

	"github.com/apache/submarine/submarine-cloud/pkg/config"
	"github.com/apache/submarine/submarine-cloud/pkg/submarine"
)

// NewSubmarineAdmin builds and returns new submarine.Admin from the list of pods
func NewSubmarineAdmin(pods []*apiv1.Pod, cfg *config.Submarine) (submarine.AdminInterface, error) {
	nodesAddrs := []string{}
	for _, pod := range pods {
		submarinePort := submarine.DefaultSubmarinePort
		glog.Info("pod = %v", pod)
		for _, container := range pod.Spec.Containers {
			if container.Name == "submarine-node" {
				for _, port := range container.Ports {
					if port.Name == "submarine" {
						submarinePort = fmt.Sprintf("%d", port.ContainerPort)
					}
				}
			}
		}
		nodesAddrs = append(nodesAddrs, net.JoinHostPort(pod.Status.PodIP, submarinePort))
	}
	adminConfig := submarine.AdminOptions{
		ConnectionTimeout:  time.Duration(cfg.DialTimeout) * time.Millisecond,
		RenameCommandsFile: cfg.GetRenameCommandsFile(),
	}

	return submarine.NewAdmin(nodesAddrs, &adminConfig), nil
}

// IsPodReady check if pod is in ready condition, return the error message otherwise
func IsPodReady(pod *apiv1.Pod) (bool, error) {
	if pod == nil {
		return false, errors.New("No Pod")
	}

	// get ready condition
	var readycondition apiv1.PodCondition
	found := false
	for _, cond := range pod.Status.Conditions {
		if cond.Type == apiv1.PodReady {
			readycondition = cond
			found = true
			break
		}
	}

	if !found {
		return false, errors.New("Cound't find ready condition")
	}

	if readycondition.Status != apiv1.ConditionTrue {
		return false, errors.New(readycondition.Message)
	}

	return true, nil
}
