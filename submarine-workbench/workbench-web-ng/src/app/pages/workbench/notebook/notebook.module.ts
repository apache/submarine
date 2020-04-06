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

import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzTabsModule } from 'ng-zorro-antd/tabs';
import { NzToolTipModule } from 'ng-zorro-antd/tooltip';

import { MonacoEditorModule } from 'ngx-monaco-editor';
import { EditorComponent } from './editor/editor.component';
import { NotebookRoutingModule } from './notebook-routing.module';
import { NotebookComponent } from './notebook.component';
import { LineChartComponent } from './result/line-chart/line-chart.component';
import { ResultComponent } from './result/result.component';
import { TableComponent } from './result/table/table.component';

@NgModule({
  declarations: [NotebookComponent, EditorComponent, ResultComponent, TableComponent, LineChartComponent],
  imports: [
    CommonModule,
    NotebookRoutingModule,
    MonacoEditorModule,
    FormsModule,
    NzIconModule,
    NzToolTipModule,
    NzTabsModule,
    NzButtonModule
  ]
})
export class NotebookModule {
}
