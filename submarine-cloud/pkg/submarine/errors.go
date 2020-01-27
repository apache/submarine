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
package submarine

import "fmt"

// Error used to represent an error
type Error string

func (e Error) Error() string { return string(e) }

// ClusterInfosError error type for submarine cluster infos access
type ClusterInfosError struct {
	errs         map[string]error
	partial      bool
	inconsistent bool
}

// nodeNotFoundedError returns when a node is not present in the cluster
const nodeNotFoundedError = Error("node not founded")

// Partial true if the some nodes of the cluster didn't answer
func (e ClusterInfosError) Partial() bool {
	return e.partial
}

// Error error string
func (e ClusterInfosError) Error() string {
	s := ""
	if e.partial {
		s += "Cluster infos partial: "
		for addr, err := range e.errs {
			s += fmt.Sprintf("%s: '%s'", addr, err)
		}
		return s
	}
	if e.inconsistent {
		s += "Cluster view is inconsistent"
	}
	return s
}

// IsPartialError returns true if the error is due to partial data recovery
func IsPartialError(err error) bool {
	e, ok := err.(ClusterInfosError)
	return ok && e.Partial()
}

// NewClusterInfosError returns an instance of cluster infos error
func NewClusterInfosError() ClusterInfosError {
	return ClusterInfosError{
		errs:         make(map[string]error),
		partial:      false,
		inconsistent: false,
	}
}
