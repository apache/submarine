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

import { Component, OnInit, ViewChild } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { ExperimentService } from '@submarine/services/experiment.service';
import { ModelVersionInfo } from '@submarine/interfaces/model-version-info';
import { ModelVersionService } from '@submarine/services/model-version.service';
import { isThisSecond } from 'date-fns';

@Component({
  selector: 'submarine-model-version',
  templateUrl: './model-version.component.html',
  styleUrls: ['./model-version.component.scss'],
})
export class ModelVersionComponent implements OnInit {
  isLoading = true;
  modelName;
  modelVersion;
  modelVersionInfo: ModelVersionInfo;
  test;

  constructor(
    private router: Router, 
    private route: ActivatedRoute, 
    private experimentService: ExperimentService,
    private modelVersionService: ModelVersionService
  ) {}

  ngOnInit() {
    this.modelName = this.route.snapshot.params.name;
    this.modelVersion = this.route.snapshot.params.version;
    this.test = 'test message';
    this.modelVersionInfo = {
      'name': "register",
      'version': "1",
      'source': "s3://submarine/experiment-1637939541827-0001/example/1",
      'userId': "test_userId",
      'experimentId': "experiment-1637939541827-0001",
      'currentStage': "None",
      'creationTime': "2021-11-26 15:12:29",
      'lastUpdatedTime': "2021-11-26 15:12:29",
      'dataset': null,
      'description': null,
      'tags': "123"
    }
    this.experimentService.emitInfo(this.modelName);
    console.log(this.modelVersionInfo);
    /**
    this.modelVersionService.querySpecificModel(this.modelName, this.modelVersion).subscribe(
      (item) => {
        this.modelVersionInfo = item;
      }
    )*/
  }
}
