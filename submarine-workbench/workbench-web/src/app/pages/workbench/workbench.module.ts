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
import { WorkbenchRoutingModule } from '@submarine/pages/workbench/workbench-routing.module';
import { PipeSharedModule } from '@submarine/pipe/pipe-shared.module';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { WorkspaceModule } from './workspace/workspace.module';
import { ExperimentModule } from './experiment/experiment.module';
import { InterpreterModule } from './interpreter/interpreter.module';
import { NotebookModule } from './notebook/notebook.module';

import { HomeComponent } from './home/home.component';
import { WorkbenchComponent } from './workbench.component';
import { WorkspaceComponent } from './workspace/workspace.component';
import { DataComponent } from './data/data.component';
import { EnvironmentModule } from './environment/environment.module';
import { TemplateModule } from './template/template.module';
import { ModelModule } from './model/model.module';

import { TranslateModule } from '@ngx-translate/core';
import TRANSLATE_CONFIG from "@submarine/core/local-translate";

@NgModule({
  declarations: [WorkbenchComponent, HomeComponent, WorkspaceComponent, DataComponent],
  imports: [
    CommonModule,
    WorkbenchRoutingModule,
    NgZorroAntdModule,
    RouterModule,
    FormsModule,
    ReactiveFormsModule,
    WorkspaceModule,
    ExperimentModule,
    InterpreterModule,
    PipeSharedModule,
    NotebookModule,
    EnvironmentModule,
    TemplateModule,
    ModelModule,
    TranslateModule.forRoot(TRANSLATE_CONFIG)
  ],
})
export class WorkbenchModule {}
