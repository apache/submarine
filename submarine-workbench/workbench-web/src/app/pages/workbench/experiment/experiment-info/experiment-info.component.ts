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

import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { ExperimentInfo } from '@submarine/interfaces/experiment-info';
import { ExperimentService } from '@submarine/services/experiment.service';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-experiment-info',
  templateUrl: './experiment-info.component.html',
  styleUrls: ['./experiment-info.component.scss'],
})
export class ExperimentInfoComponent implements OnInit {
  isLoading = true;
  experimentID;
  experimentInfo: ExperimentInfo;
  currentState = 0;
  selectedPod;
  podNameArr;
  podLogArr;
  paramData;
  metricData;
  artifactPaths;

  constructor(
    private router: Router,
    private route: ActivatedRoute,
    private experimentService: ExperimentService,
    private nzMessageService: NzMessageService
  ) {}

  statusColor: { [key: string]: string } = {
    Accepted: 'gold',
    Created: 'white',
    Running: 'green',
    Succeeded: 'blue',
  };

  ngOnInit() {
    this.experimentID = this.route.snapshot.params.id;
    this.experimentService.querySpecificExperiment(this.experimentID).subscribe(
      (item) => {
        this.experimentInfo = item;
        this.isLoading = false;
        if (this.experimentInfo.status === 'Succeeded') {
          const finTime = new Date(this.experimentInfo.finishedTime);
          const runTime = new Date(this.experimentInfo.runningTime);
          const result = (finTime.getTime() - runTime.getTime()) / 1000;
          this.experimentInfo.duration = this.experimentService.durationHandle(result);
        } else {
          const currentTime = new Date();
          const runTime = new Date(this.experimentInfo.runningTime);
          const result = (currentTime.getTime() - runTime.getTime()) / 1000;
          this.experimentInfo.duration = this.experimentService.durationHandle(result);
        }
      },
      (err) => {
        this.nzMessageService.error('Cannot load ' + this.experimentID);
        this.router.navigate(['/workbench/experiment']);
      }
    );

    this.getExperimentPod();
    this.getExperimentArtifactPaths();
    this.experimentService.emitInfo(this.experimentID);
  }

  getExperimentPod() {
    this.experimentService.getExperimentLog(this.experimentID).subscribe(
      (result) => {
        this.podNameArr = result.logContent.map((item) => Object.values(item)[0]);
        this.selectedPod = this.podNameArr[0];
        this.podLogArr = result.logContent;
      },
      (err) => {
        this.nzMessageService.error('Cannot load pod of ' + this.experimentID);
        console.log(err);
      }
    );

    this.experimentService
      .getExperimentParam({
        id: this.experimentID,
      })
      .subscribe(
        (result) => {
          this.paramData = result;
        },
        (err) => {
          this.nzMessageService.error('Cannot load param of ' + this.experimentID);
          console.log(err);
        }
      );

    this.experimentService
      .getExperimentMetric({
        id: this.experimentID,
      })
      .subscribe(
        (result) => {
          this.metricData = result;
        },
        (err) => {
          this.nzMessageService.error('Cannot load metric of ' + this.experimentID);
          console.log(err);
        }
      );
  }

  getExperimentArtifactPaths() {
    this.experimentService
      .getExperimentArtifactPaths(this.experimentID)
      .subscribe(
        (result) => {
          this.artifactPaths = result;
        },
        (err) => {
          this.nzMessageService.error('Cannot load artifact paths of ' + this.experimentID);
          console.log(err);
        }
      );
  }

  onDeleteExperiment() {
    this.experimentService.deleteExperiment(this.experimentID).subscribe(
      () => {
        this.nzMessageService.success('Delete experiment success!');
        this.router.navigate(['/workbench/experiment']);
      },
      (err) => {
        this.nzMessageService.success(err.message);
      }
    );
  }

  // TODO(jasoonn): Start experiment
  startExperiment() {}
  // TODO(jasoonn): Edit experiment
  editExperiment() {}
}
