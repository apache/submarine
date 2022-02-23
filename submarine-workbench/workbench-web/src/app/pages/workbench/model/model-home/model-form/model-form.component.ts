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

import { Component, OnInit, Output, EventEmitter, Input } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ModelInfo } from '@submarine/interfaces/model-info';
import { ModelVersionInfo } from '@submarine/interfaces/model-version-info';
import { ModelVersionService } from '@submarine/services/model-version.service';
import { ModelService } from '@submarine/services/model.service';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
    selector: 'model-form',
    templateUrl: './model-form.component.html',
    styleUrls: ['./model-form.component.scss']
})
export class ModelFormComponent implements OnInit {
  isVisible: boolean;
  modelForm: FormGroup;

  registeredModelList = []

  constructor(
    private fb: FormBuilder,
    private modelService: ModelService,
    private nzMessageService: NzMessageService,
  ) {}

  ngOnInit(): void {
    this.modelForm = this.fb.group({
      modelName: [null, Validators.required],
      description: [null],
      tags: this.fb.array([]),
    });
    this.fetchRegisteredModelList();
  }

  get modelName() {
    return this.modelForm.get("modelName");
  }

  get description() {
    return this.modelForm.get("description");
  }

  get tags(){
    return this.modelForm.get("tags");
  }

  initModal() {
      this.isVisible = true;
      console.log("what?");
  }

  closeModal(){
    this.modelForm.reset();
    this.isVisible = false;
  }

  fetchRegisteredModelList(){
    this.modelService.fetchModelList().subscribe(list => {
      list.forEach(e => {
        this.registeredModelList.push(e.name);
      })
    })
  }

  checkStatus() {
    return this.modelName.invalid;
  }

  submitForm() {
    const modelInfo : ModelInfo = {
      name: this.modelName.value,
      creationTime: null,
      lastUpdatedTime: null,
      description: this.description.value,
      tags: this.tags.value,
    }
    this.modelService.createModel(modelInfo).subscribe({
      next: (result) => {
        console.log(result)
        this.nzMessageService.success('Create Model Registry Success!');
        this.closeModal();
      },
      error: (msg) => {
        console.log(msg)
        this.nzMessageService.error(`Model registry with name: ${modelInfo.name} is exist.`, {
          nzPauseOnHover: true,
        });
      },
    })
    // this.modelVersionService.createModelVersion(modelVersionInfo, this.reviseBaseDir(this.baseDir)).subscribe({
    //   next: (result) => {
    //     this.nzMessageService.success('Register Model Success!');
    //     this.closeModal();
    //   },
    //   error: (msg) => {
    //     console.log(msg)
    //     this.nzMessageService.error(`Current model has been registered in this registered model.`, {
    //       nzPauseOnHover: true,
    //     });
    //   },
    // })
  }
}