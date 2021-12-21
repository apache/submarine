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

import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { ModelRoutingModule } from './model-routing.module';
import { RouterModule } from '@angular/router';
import { ModelComponent } from './model.component';
import { ModelVersionComponent } from './model-version/model-version.component';
import { PipeSharedModule } from '@submarine/pipe/pipe-shared.module';
import { ModelHomeComponent } from './model-home/model-home.component';
import { ModelCardsComponent } from './model-home/model-cards/model-cards.component';
import { ModelCardComponent } from './model-home/model-cards/model-card/model-card.component';
import { ModelTagComponent } from './model-tag/model-tag.component';
import { ModelInfoComponent } from './model-info/model-info.component';

@NgModule({
  declarations: [
    ModelComponent,
    ModelHomeComponent,
    ModelVersionComponent,
    ModelCardsComponent,
    ModelCardComponent,
    ModelTagComponent,
    ModelInfoComponent,
  ],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    NgZorroAntdModule,
    RouterModule,
    ModelRoutingModule,
    PipeSharedModule,
  ],
  providers: [],
  exports: [ModelComponent],
})
export class ModelModule {}
