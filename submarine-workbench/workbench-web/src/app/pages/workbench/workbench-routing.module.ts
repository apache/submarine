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
import { ActivatedRouteSnapshot, RouterModule, RouterStateSnapshot, Routes } from '@angular/router';
import { EnvironmentComponent } from '@submarine/pages/workbench/environment/environment.component';
import { ExperimentComponent } from '@submarine/pages/workbench/experiment/experiment.component';
import { WorkbenchComponent } from '@submarine/pages/workbench/workbench.component';
import { DataComponent } from './data/data.component';
import { ExperimentInfoComponent } from './experiment/experiment-info/experiment-info.component';
import { HomeComponent } from './home/home.component';
import { InterpreterComponent } from './interpreter/interpreter.component';
import { ModelComponent } from './model/model.component';
import { NotebookComponent } from './notebook/notebook.component';
import { WorkspaceComponent } from './workspace/workspace.component';

const routes: Routes = [
  {
    path: '',
    component: WorkbenchComponent,
    children: [
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'experiment'
      },
      {
        path: 'home',
        component: HomeComponent,
        canActivate: ['canActivatePage']
      },
      {
        path: 'workspace',
        component: WorkspaceComponent,
        canActivate: ['canActivatePage']
      },
      {
        path: 'interpreter',
        component: InterpreterComponent,
        canActivate: ['canActivatePage']
      },
      {
        path: 'experiment',
        component: ExperimentComponent,
        children: [
          {
            path: 'info/:id',
            component: ExperimentInfoComponent
          }
        ],
        canActivate: ['canActivatePage'],
        canActivateChild: ['canActivatePage']
      },
      {
        path: 'environment',
        component: EnvironmentComponent,
        canActivate: ['canActivatePage']
      },
      {
        path: 'data',
        component: DataComponent,
        canActivate: ['canActivatePage']
      },
      {
        path: 'model',
        component: ModelComponent,
        canActivate: ['canActivatePage']
      },
      {
        path: 'notebook',
        component: NotebookComponent,
        canActivate: ['canActivatePage'],
      },
      {
        path: 'manager',
        loadChildren: () => import('./manager/manager.module').then((m) => m.ManagerModule),
        canActivate: ['canActivatePage']
      }
    ]
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  providers: [
    {
      provide: 'canActivatePage',
      useValue: (route: ActivatedRouteSnapshot, state: RouterStateSnapshot) => {
        const disablePaths = ['home', 'data', 'model', 'workspace', 'interpreter'];
        let currentPage = state.url.split('/')[2];
        console.log('currentPage', currentPage);
        if (disablePaths.includes(currentPage)) return false;
        else return true;
      }
    }
  ]
})
export class WorkbenchRoutingModule { }
