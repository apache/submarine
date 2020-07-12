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

import { Component, OnInit, Input, SimpleChanges } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BaseApiService } from '@submarine/services/base-api.service';

@Component({
  selector: 'submarine-hyper-params',
  templateUrl: './hyper-params.component.html',
  styleUrls: ['./hyper-params.component.scss']
})
export class HyperParamsComponent implements OnInit {
  @Input() workerIndex;
  @Input() paramData;
  podParam = [];

  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  ngOnInit() {}

  ngOnChanges(chg: SimpleChanges) {
    this.paramData.forEach((data) => {
      if (data.workerIndex == this.workerIndex) {
        this.podParam.push(data);
      }
    });
  }
}
