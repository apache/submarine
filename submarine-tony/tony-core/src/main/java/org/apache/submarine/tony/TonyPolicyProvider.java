/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package org.apache.submarine.tony;

import org.apache.submarine.tony.rpc.MetricsRpc;
import org.apache.submarine.tony.rpc.TensorFlowCluster;
import org.apache.hadoop.security.authorize.PolicyProvider;
import org.apache.hadoop.security.authorize.Service;

/**
 * PolicyProvider for Client to AM protocol.
 **/
public class TonyPolicyProvider extends PolicyProvider {
    @Override
    public Service[] getServices() {
        return new Service[]{
            new Service("tony.cluster", TensorFlowCluster.class),
            new Service("tony.metrics", MetricsRpc.class)
        };
    }
}
