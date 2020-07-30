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
  styleUrls: ['./experiment-info.component.scss']
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

  constructor(
    private router: Router,
    private route: ActivatedRoute,
    private experimentService: ExperimentService,
    private nzMessageService: NzMessageService
  ) {}

  ngOnInit() {
    this.experimentID = this.route.snapshot.params.id;
    this.experimentService.querySpecificExperiment(this.experimentID).subscribe(
      (item) => {
        this.experimentInfo = item;
        this.isLoading = false;
        if (this.experimentInfo.status == 'Succeeded') {
          var finTime = new Date(this.experimentInfo.finishedTime);
          var runTime = new Date(this.experimentInfo.runningTime);
          var result = (finTime.getTime() - runTime.getTime()) / 1000;
          this.experimentInfo.duration = this.durationHandle(result);
        } else {
          var currentTime = new Date();
          var runTime = new Date(this.experimentInfo.runningTime);
          var result = (currentTime.getTime() - runTime.getTime()) / 1000;
          this.experimentInfo.duration = this.durationHandle(result);
        }
      },
      (err) => {
        this.nzMessageService.error('Cannot load ' + this.experimentID);
        this.router.navigate(['/workbench/experiment']);
      }
    );

    this.getExperimentPod();
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
        id: this.experimentID
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
        id: this.experimentID
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

  durationHandle(secs: number) {
    var hr = Math.floor(secs / 3600);
    var min = Math.floor((secs - hr * 3600) / 60);
    var sec = Math.round(secs) - hr * 3600 - min * 60;
    var showHr;
    var showMin;
    var showSec;
    if (hr < 10) {
      showHr = '0' + hr;
    } else {
      showHr = hr.toString();
    }
    if (min < 10) {
      showMin = '0' + min;
    } else {
      showMin = min.toString();
    }
    if (sec < 10) {
      showSec = '0' + sec;
    } else {
      showSec = sec.toString();
    }
    return showHr + ':' + showMin + ':' + showSec;
  }

  // TODO(jasoonn): Start experiment
  startExperiment() {}
  // TODO(jasoonn): Edit experiment
  editExperiment() {}
}
