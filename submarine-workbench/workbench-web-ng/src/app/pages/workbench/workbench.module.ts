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
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { WorkbenchRoutingModule } from '@submarine/pages/workbench/workbench-routing.module';
import { PipeSharedModule } from '@submarine/pipe/pipe-shared.module';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { DataComponent } from './data/data.component';
import { ExperimentComponent } from './experiment/experiment.component';
import { ExperimentModule } from './experiment/experiment.module';

import { HomeComponent } from './home/home.component';
import { InterpreterModule } from './interpreter/interpreter.module';
import { ModelComponent } from './model/model.component';
import { WorkbenchComponent } from './workbench.component';
import { WorkspaceComponent } from './workspace/workspace.component';
import { WorkspaceModule } from './workspace/workspace.module';
import { EnvironmentComponent } from './environment/environment.component';

@NgModule({
  declarations: [
    WorkbenchComponent,
    HomeComponent,
    WorkspaceComponent,
    ExperimentComponent,
    DataComponent,
    ModelComponent,
    EnvironmentComponent
  ],
  imports: [
    CommonModule,
    WorkbenchRoutingModule,
    NgZorroAntdModule,
    RouterModule,
    FormsModule,
    WorkspaceModule,
    ExperimentModule,
    InterpreterModule,
    PipeSharedModule
  ]
})
export class WorkbenchModule {}
