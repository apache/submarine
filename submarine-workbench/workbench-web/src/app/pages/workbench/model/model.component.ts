/*!
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
import { ModelService } from '@submarine/services/model.service';
import { ModelVersionService } from '@submarine/services/model-version.service';
import { delay } from 'rxjs/operators';

@Component({
  selector: 'submarine-model',
  templateUrl: './model.component.html',
  styleUrls: ['./model.component.scss']
})
export class ModelComponent implements OnInit {
  modelName: string = null;
  modelVersion: string = null;

  constructor(private modelService: ModelService, private modelVersionService: ModelVersionService) {}

  ngOnInit() {
    this.modelService.infoEmitted$.pipe(delay(0)).subscribe((name) => (this.modelName = name));
    this.modelVersionService.infoEmitted$.pipe(delay(0)).subscribe((version) => this.modelVersion = version);
  }
}
