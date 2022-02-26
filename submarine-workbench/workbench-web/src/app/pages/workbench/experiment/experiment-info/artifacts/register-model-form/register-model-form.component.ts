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
import { ModelVersionInfo } from '@submarine/interfaces/model-version-info';
import { ModelVersionService } from '@submarine/services/model-version.service';
import { ModelService } from '@submarine/services/model.service';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
    selector: 'register-model-form',
    templateUrl: './register-model-form.component.html',
    styleUrls: ['./register-model-form.component.scss']
})
export class RegisterModelFormComponent implements OnInit {
  @Input()  baseDir : string;
  @Input() experimentId : string;
  
  isVisible: boolean;
  registerModelForm: FormGroup;

  registeredModelList = []

  constructor(
    private fb: FormBuilder,
    private modelService: ModelService,
    private modelVersionService: ModelVersionService,
    private nzMessageService: NzMessageService,
  ) {}

  ngOnInit(): void {
    this.registerModelForm = this.fb.group({
      registeredModelName: [null, Validators.required],
      description: [null],
      tags: this.fb.array([]),
    });
    this.fetchRegisteredModelList();
  }

  get registeredModelName() {
    return this.registerModelForm.get("registeredModelName");
  }

  get description() {
    return this.registerModelForm.get("description");
  }

  get tags(){
    return this.registerModelForm.get("tags");
  }

  initModal() {
      this.isVisible = true;
  }

  closeModal(){
    this.registerModelForm.reset();
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
    return this.registeredModelName.invalid;
  }

  submitForm() {
    const modelVersionInfo : ModelVersionInfo = {
      name: this.registeredModelName.value,
      version: null,
      userId: "",                 // TODO(KUAN-HSUN-LI) use specific user name
      experimentId: this.experimentId,
      currentStage: "None",
      creationTime: null,
      lastUpdatedTime: null,
      dataset: null,
      description: this.description.value,
      tags: this.tags.value,
    }
    this.modelVersionService.createModelVersion(modelVersionInfo, this.reviseBaseDir(this.baseDir)).subscribe({
      next: (result) => {
        this.nzMessageService.success('Register Model Success!');
        this.closeModal();
      },
      error: (msg) => {
        this.nzMessageService.error(`Current model has been registered in this registered model.`, {
          nzPauseOnHover: true,
        });
      },
    })
  }

  reviseBaseDir(baseDir: string) : string {
    // slice the "s3://submarine/" characters and the last '/'
    return baseDir.slice(15, baseDir.length-1);
  }

}