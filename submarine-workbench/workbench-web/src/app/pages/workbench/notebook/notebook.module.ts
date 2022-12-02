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

import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { PipeSharedModule } from '@submarine/pipe/pipe-shared.module';
import { NotebookService } from '@submarine/services/notebook-services/notebook.service';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { NotebookRoutingModule } from './notebook-routing.module';

import { NotebookComponent } from './notebook.component';
import { NotebookHomeComponent } from './notebook-home/notebook-home.component';
import { NotebookListComponent } from './notebook-home/notebook-list/notebook-list.component';
import { NotebookFormComponent } from './notebook-home/notebook-form/notebook-form.component';

import { TranslateModule } from '@ngx-translate/core';
import TRANSLATE_CONFIG from "@submarine/core/local-translate";

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    RouterModule,
    PipeSharedModule,
    NgZorroAntdModule,
    NotebookRoutingModule,
    TranslateModule.forChild(TRANSLATE_CONFIG)
  ],
  providers: [NotebookService],
  declarations: [NotebookComponent, NotebookHomeComponent, NotebookListComponent, NotebookFormComponent],
  exports: [NotebookComponent],
})
export class NotebookModule {}
