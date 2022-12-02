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
import { ModelService } from '@submarine/services/model.service';
import { NzMessageService } from 'ng-zorro-antd';
import { TranslateService } from '@ngx-translate/core';

@Component({
    selector: 'model-form',
    templateUrl: './model-form.component.html',
    styleUrls: ['./model-form.component.scss']
})
export class ModelFormComponent implements OnInit {

  @Input() fetchModelCards;

  isVisible: boolean;
  modelForm: FormGroup;

  registeredModelList = []

  constructor(
    private fb: FormBuilder,
    private modelService: ModelService,
    private nzMessageService: NzMessageService,
    private translate: TranslateService
  ) {
  }

  ngOnInit(): void {
    this.modelForm = this.fb.group({
      modelName: [null, Validators.required],
      description: [null],
      tags: this.fb.array([])
    });
    this.fetchRegisteredModelList();
  }

  get modelName() {
    return this.modelForm.get("modelName");
  }

  get description() {
    return this.modelForm.get("description");
  }

  get tags() {
    return this.modelForm.get("tags");
  }

  initModal() {
      this.isVisible = true;
  }

  closeModal() {
    this.modelForm.reset();
    this.isVisible = false;
  }

  fetchRegisteredModelList() {
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
    const modelInfo: ModelInfo = {
      name: this.modelName.value,
      creationTime: null,
      lastUpdatedTime: null,
      description: this.description.value,
      tags: this.tags.value
    }
    this.modelService.createModel(modelInfo).subscribe({
      next: (result) => {
        this.nzMessageService.success(this.translate.instant('Create Model Registry Success!'));
        this.closeModal();
        // refresh model card list
        this.fetchModelCards();
      },
      error: (msg) => {
        this.nzMessageService.error(this.translate.instant('Model registry with name') + `: ${modelInfo.name} ` + this.translate.instant('is exist.'), {
          nzPauseOnHover: true
        });
      }
    })
  }
}
