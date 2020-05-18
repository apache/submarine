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
import { RouterModule, Routes } from '@angular/router';
import { ExperimentComponent } from '@submarine/pages/workbench/experiment/experiment.component';
import { WorkbenchComponent } from '@submarine/pages/workbench/workbench.component';
import { DataComponent } from './data/data.component';
import { HomeComponent } from './home/home.component';
import { InterpreterComponent } from './interpreter/interpreter.component';
import { ModelComponent } from './model/model.component';
import { WorkspaceComponent } from './workspace/workspace.component';

const routes: Routes = [
  {
    path: '',
    component: WorkbenchComponent,
    children: [
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'home'
      },
      {
        path: 'home',
        component: HomeComponent
      },
      {
        path: 'workspace',
        component: WorkspaceComponent
      },
      {
        path: 'interpreter',
        component: InterpreterComponent
      },
      {
        path: 'experiment',
        component: ExperimentComponent
      },
      {
        path: 'data',
        component: DataComponent
      },
      {
        path: 'model',
        component: ModelComponent
      },
      {
        path: 'manager',
        loadChildren: () => import('./manager/manager.module').then((m) => m.ManagerModule)
      }
    ]
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)]
})
export class WorkbenchRoutingModule {}
