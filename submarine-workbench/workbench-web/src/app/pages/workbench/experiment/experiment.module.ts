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
import { ExperimentService } from '@submarine/services/experiment.service';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { ModelModule } from '../model/model.module';
import { ExperimentCustomizedFormComponent } from './experiment-home/experiment-form/experiment-customized-form/experiment-customized-form.component';
import { ExperimentFormComponent } from './experiment-home/experiment-form/experiment-form.component';
import { ExperimentPredefinedFormComponent } from './experiment-home/experiment-form/experiment-predefined-form/experiment-predefined-form.component';
import { ExperimentHomeComponent } from './experiment-home/experiment-home.component';
import { ExperimentListComponent } from './experiment-home/experiment-list/experiment-list.component';
import { ArtifactsComponent } from './experiment-info/artifacts/artifacts.component';
import { RegisterModelFormComponent } from './experiment-info/artifacts/register-model-form/register-model-form.component';
import { RegisterModelTagsComponent } from './experiment-info/artifacts/register-model-form/register-model-tag/register-model-tags.component';
import { ChartsComponent } from './experiment-info/charts/charts.component';
import { ExperimentInfoComponent } from './experiment-info/experiment-info.component';
import { HyperParamsComponent } from './experiment-info/hyper-params/hyper-params.component';
import { MetricsComponent } from './experiment-info/metrics/metrics.component';
import { OutputsComponent } from './experiment-info/outputs/outputs.component';
import { ExperimentRoutingModule } from './experiment-routing.module';
import { ExperimentComponent } from './experiment.component';

import { TranslateModule } from '@ngx-translate/core';
import TRANSLATE_CONFIG from "@submarine/core/local-translate";

@NgModule({
  exports: [ExperimentComponent],
  imports: [
    ReactiveFormsModule,
    NgZorroAntdModule,
    CommonModule,
    FormsModule,
    NgxChartsModule,
    RouterModule,
    PipeSharedModule,
    ExperimentRoutingModule,
    ModelModule,
    TranslateModule.forChild(TRANSLATE_CONFIG)
  ],
  providers: [ExperimentService],
  declarations: [
    ExperimentComponent,
    ExperimentListComponent,
    ExperimentInfoComponent,
    HyperParamsComponent,
    MetricsComponent,
    ChartsComponent,
    OutputsComponent,
    ArtifactsComponent,
    ExperimentHomeComponent,
    ExperimentFormComponent,
    ExperimentPredefinedFormComponent,
    ExperimentCustomizedFormComponent,
    RegisterModelFormComponent,
    RegisterModelTagsComponent,
  ],
})
export class ExperimentModule {}
