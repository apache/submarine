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
import { EnvironmentRoutingModule } from './environment-routing.module';
import { RouterModule } from '@angular/router';
import { EnvironmentHomeComponent } from './environment-home/environment-home.component';
import { EnvironmentService } from '@submarine/services/environment-services/environment.service';
import { EnvironmentComponent } from './environment.component';
import { EnvironmentListComponent } from './environment-home/environment-list/environment-list.component';
import { EnvironmentFormComponent } from './environment-home/environment-form/environment-form.component';

import { TranslateModule } from '@ngx-translate/core';
import TRANSLATE_CONFIG from "@submarine/core/local-translate";

@NgModule({
  declarations: [EnvironmentComponent, EnvironmentHomeComponent, EnvironmentListComponent, EnvironmentFormComponent],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    FormsModule,
    RouterModule,
    NgZorroAntdModule,
    EnvironmentRoutingModule,
    TranslateModule.forChild(TRANSLATE_CONFIG)
  ],
  providers: [EnvironmentService],
  exports: [EnvironmentComponent],
})
export class EnvironmentModule {}
