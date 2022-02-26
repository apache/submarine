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

import { Component, Input, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { ModelVersionService } from '@submarine/services/model-version.service';
import { ModelService } from '@submarine/services/model.service';
import { ModelInfo } from '@submarine/interfaces/model-info';
import { ModelVersionInfo } from '@submarine/interfaces/model-version-info';
import {humanizeTime} from '@submarine/pages/workbench/utils/humanize-time'
import { ModelServeService } from '@submarine/services/model-serve.service';
import { NzMessageService } from 'ng-zorro-antd/message';


@Component({
  selector: 'submarine-model-info',
  templateUrl: './model-info.component.html',
  styleUrls: ['./model-info.component.scss'],
})
export class ModelInfoComponent implements OnInit {
  isModelInfoLoading: boolean = true;
  isModelVersionsLoading: boolean = true;
  modelName: string;
  selectedModelInfo: ModelInfo; 
  modelVersions: ModelVersionInfo[];
  humanizedCreationTime: string;
  humanizedLastUpdatedTime: string;

  constructor(
    private router: Router, 
    private route: ActivatedRoute, 
    private modelVersionService: ModelVersionService, 
    private modelService: ModelService,
    private modelServeService: ModelServeService,
    private nzMessageService: NzMessageService,
  ) {}

  ngOnInit(): void {
    this.modelName = this.route.snapshot.params.name;
    this.modelService.emitInfo(this.modelName);
    this.fetchSpecificRegisteredModel();
    this.fetchModelAllVersions();
  }

  fetchSpecificRegisteredModel = () => {
    this.modelService.querySpecificModel(this.modelName).subscribe(
      (res) => {
        this.selectedModelInfo = res;
        this.humanizedCreationTime = humanizeTime(res.creationTime);
        this.humanizedLastUpdatedTime = humanizeTime(res.lastUpdatedTime);
        this.isModelInfoLoading = false;
      }
    )
  }

  fetchModelAllVersions = () => {
    this.modelVersionService.queryModelAllVersions(this.modelName).subscribe(
      (res) => {
        this.modelVersions = res;
        this.isModelVersionsLoading = false;
      }
    );
  }

  onCreateServe = (version: number) => {
    this.modelServeService.createServe(this.modelName, version).subscribe({
      next: (result) => {
        this.nzMessageService.success(`The model serve with name: ${this.modelName} and version: ${version} is created.`)
      },
      error: (msg) => {
        this.nzMessageService.error(`${msg}, please try again`, {
          nzPauseOnHover: true,
        });
      },
    })
  }

  onDeleteServe = (version: number) => {
    this.modelServeService.deleteServe(this.modelName, version).subscribe({
      next: (result) => {
        this.nzMessageService.success(`The model serve with name: ${this.modelName} and version: ${version} is deleted.`)
      },
      error: (msg) => {
        this.nzMessageService.error(`${msg}, please try again`, {
          nzPauseOnHover: true,
        });
      },
    })
  }

  onDeleteModelVersion = (version:number) => {
    this.modelVersionService.deleteModelVersion(this.modelName, version).subscribe({
      next: (result) => {
        this.nzMessageService.success(`The model with name: ${this.modelName} and version: ${version} is deleted.`)
      },
      error: (msg) => {
        this.nzMessageService.error(`${msg}, please try again`, {
          nzPauseOnHover: true,
        });
      },
    })
  }

  preventEvent(e){
    e.preventDefault();
    e.stopPropagation();
  }
}

