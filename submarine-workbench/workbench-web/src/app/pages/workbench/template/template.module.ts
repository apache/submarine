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
import { TemplateRoutingModule } from './template-routing.module';
import { RouterModule } from '@angular/router';
import { TemplateHomeComponent } from './template-home/template-home.component';
import { TemplateFormComponent } from './template-home/template-form/template-form.component';
import { TemplateListComponent } from './template-home/template-list/template-list.component';
import { TemplateComponent } from './template.component';
import { TemplateInfoComponent } from './template-info/template-info.component';
import { PipeSharedModule } from '@submarine/pipe/pipe-shared.module';

import { TranslateModule } from '@ngx-translate/core';
import TRANSLATE_CONFIG from "@submarine/core/local-translate";

@NgModule({
  declarations: [
    TemplateComponent,
    TemplateHomeComponent,
    TemplateFormComponent,
    TemplateListComponent,
    TemplateInfoComponent,
  ],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    NgZorroAntdModule,
    RouterModule,
    TemplateRoutingModule,
    PipeSharedModule,
    TranslateModule.forChild(TRANSLATE_CONFIG)
  ],
  providers: [],
  exports: [TemplateComponent],
})
export class TemplateModule {}
