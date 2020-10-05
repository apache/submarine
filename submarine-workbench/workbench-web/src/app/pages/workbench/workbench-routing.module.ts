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
import { EnvironmentComponent } from '@submarine/pages/workbench/environment/environment.component';
import { ExperimentComponent } from '@submarine/pages/workbench/experiment/experiment.component';
import { WorkbenchComponent } from '@submarine/pages/workbench/workbench.component';
import { DataComponent } from './data/data.component';
import { ExperimentInfoComponent } from './experiment/experiment-info/experiment-info.component';
import { HomeComponent } from './home/home.component';
import { InterpreterComponent } from './interpreter/interpreter.component';
import { ModelComponent } from './model/model.component';
import { WorkspaceComponent } from './workspace/workspace.component';

function disablePage(allRoutes: Routes): Routes {
  const disabledList: string[] = ['home', 'data', 'model', 'workspace', 'interpreter'];
  allRoutes[0].children[0].redirectTo = 'experiment'; // redirect root page to experiment
  allRoutes[0].children = allRoutes[0].children.filter((item) => !disabledList.includes(item.path)); // filter pages which are incomplete
  return allRoutes;
}

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
        component: ExperimentComponent,
        children: [
          {
            path: 'info/:id',
            component: ExperimentInfoComponent
          }
        ]
      },
      {
        path: 'environment',
        component: EnvironmentComponent
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
      },
      {
        path: 'notebook',
        loadChildren: () => import('./notebook/notebook.module').then((m) => m.NotebookModule)
      }
    ]
  }
];

@NgModule({
  imports: [RouterModule.forChild(disablePage(routes))]
})
export class WorkbenchRoutingModule {}
