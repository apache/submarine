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

import { HttpErrorResponse } from '@angular/common/http';
import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { ModelInfo } from '@submarine/interfaces/model-info';
import { ModelVersionService } from '@submarine/services/model-version.service';
import { ModelService } from '@submarine/services/model.service';
import { NzMessageService } from 'ng-zorro-antd';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'submarine-model-card',
  templateUrl: './model-card.component.html',
  styleUrls: ['./model-card.component.scss'],
})
export class ModelCardComponent implements OnInit {
  @Input() card: ModelInfo;
  // tslint:disable-next-line:prefer-output-readonly
  @Output() private refreshCards = new EventEmitter<boolean>();
  description: string;

  constructor(
    private modelService: ModelService,
    private modelVersionService: ModelVersionService,
    private nzMessageService: NzMessageService,
    private translate: TranslateService
    ) {
    }

  ngOnInit() {
    if (this.card.description && this.card.description.length > 15) {
      this.description = this.card.description.substring(0,50) + "...";
    }
    else {
      this.description = this.card.description;
    }
  }

  onDeleteModelRegistry(modelName: string){
    this.modelService.deleteModel(modelName).subscribe({
      next: (result) => {
        this.nzMessageService.success(this.translate.instant('Delete registered model success!'));
        // send EventEmitter true to refresh cards
        this.refreshCards.emit(true)
      },
      error: (err: HttpErrorResponse) => {
        this.nzMessageService.error(`${err.error.message}`, {
          nzPauseOnHover: true,
        });
      },
    })
  }

  preventEvent(e) {
    e.preventDefault();
    e.stopPropagation();
  }
}
