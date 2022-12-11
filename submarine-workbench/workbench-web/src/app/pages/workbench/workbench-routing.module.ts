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
import { WorkbenchComponent } from '@submarine/pages/workbench/workbench.component';
import { DataComponent } from './data/data.component';
import { HomeComponent } from './home/home.component';
import { InterpreterComponent } from './interpreter/interpreter.component';
import { WorkspaceComponent } from './workspace/workspace.component';

const routes: Routes = [
  {
    path: '',
    component: WorkbenchComponent,
    children: [
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'experiment',
      },
      {
        path: 'home',
        component: HomeComponent,
        canActivate: ['canActivatePage'],
      },
      {
        path: 'workspace',
        component: WorkspaceComponent,
        canActivate: ['canActivatePage'],
      },
      {
        path: 'interpreter',
        component: InterpreterComponent,
        canActivate: ['canActivatePage'],
      },
      {
        path: 'experiment',
        loadChildren: () => import('./experiment/experiment.module').then((m) => m.ExperimentModule),
        canActivate: ['canActivatePage'],
      },
      {
        path: 'environment',
        loadChildren: () => import('./environment/environment.module').then((m) => m.EnvironmentModule),
        canActivate: ['canActivatePage'],
      },
      {
        path: 'template',
        loadChildren: () => import('./template/template.module').then((m) => m.TemplateModule),
        canActivate: ['canActivatePage'],
      },
      {
        path: 'model',
        loadChildren: () => import('./model/model.module').then((m) => m.ModelModule),
        canActivate: ['canActivatePage'],
      },
      {
        path: 'data',
        component: DataComponent,
        canActivate: ['canActivatePage'],
      },
      {
        path: 'notebook',
        loadChildren: () => import('./notebook/notebook.module').then((m) => m.NotebookModule),
        canActivate: ['canActivatePage'],
      },
      {
        path: 'manager',
        loadChildren: () => import('./manager/manager.module').then((m) => m.ManagerModule),
        canActivate: ['canActivatePage'],
      },
    ],
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  providers: [
    {
      provide: 'canActivatePage',
      useValue: (route: ActivatedRouteSnapshot, state: RouterStateSnapshot) => {
        const disablePaths = ['home', 'data', 'workspace', 'interpreter'];
        let currentPage = state.url.split('/')[2];
        // console.log('currentPage', currentPage);
        if (disablePaths.includes(currentPage)) return false;
        else return true;
      },
    },
  ],
})
export class WorkbenchRoutingModule {}
