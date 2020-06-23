/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { Component, OnInit } from '@angular/core';

interface HyperParams {
  key: string;
  value: string;
}

@Component({
  selector: 'submarine-hyper-params',
  templateUrl: './hyper-params.component.html',
  styleUrls: ['./hyper-params.component.scss']
})
export class HyperParamsComponent implements OnInit {
  paramsList : HyperParams[] = []

  constructor() {}

  ngOnInit() {
    // TODO(chiajoukuo): get data from server
    this.paramsList = [
      {
        key:'conf',
        value: '/var/tf_deepfm/deepfm.json'
      },
      {
        key:'train_beta1',
        value: '0.9'
      },
      {
        key:'train_beta2',
        value: '0.999'
      },
      {
        key:'train_epsilon',
        value: '1.0E-8'
      },
      {
        key:'train_lr',
        value: '5.0E-4'
      },
      {
        key:'train_Optimizer',
        value: 'AdamOptimizer'
      }
    ]
  }
}
