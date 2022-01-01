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

import { Component, Input, OnInit, SimpleChanges } from '@angular/core';

@Component({
  selector: 'submarine-charts',
  templateUrl: './charts.component.html',
  styleUrls: ['./charts.component.scss']
})
export class ChartsComponent implements OnInit {
  @Input() workerIndex;
  @Input() metricData;

  title = 'Metrics';
  podMetrics = {};
  view: any[] = [1000, 300];
  legend: boolean = false;
  showLabels: boolean = true;
  animations: boolean = true;
  xAxis: boolean = true;
  yAxis: boolean = true;
  showYAxisLabel: boolean = true;
  showXAxisLabel: boolean = true;
  xAxisLabel: string = 'Time (s)';
  yAxisLabels = [];
  timeline: boolean = false;
  colorScheme = ['cool', 'fire', 'flame', 'air', 'forest', 'neons', 'ocean'];
  constructor() {}

  onSelect(data): void {
    console.log('Item clicked', JSON.parse(JSON.stringify(data)));
  }

  onActivate(data): void {
    console.log('Activate', JSON.parse(JSON.stringify(data)));
  }

  onDeactivate(data): void {
    console.log('Deactivate', JSON.parse(JSON.stringify(data)));
  }

  ngOnInit() {}

  ngOnChanges(chg: SimpleChanges) {
    this.podMetrics = {};
    this.yAxisLabels = [];
    this.fetchMetric();
  }
  fetchMetric() {
    if (this.metricData === undefined) {
      return;
    }
    let key = '';
    let metrics = [];
    this.metricData.forEach((data) => {
      if (data.workerIndex === undefined) {
        return;
      }
      if (this.workerIndex.indexOf(data.workerIndex) >= 0) {
        if (data.key !== key && metrics.length > 0) {
          this.yAxisLabels.push(key);
          this.podMetrics[key] = [];
          this.podMetrics[key].push({ name: key, series: metrics });
          metrics = [];
        }
        key = data.key;
        data.timestamp = data.timestamp.replace(" ", "T")
        const d = new Date(data.timestamp);
        const metric = { name: d, value: data.value };
        metrics.push(metric);
      }
    });
    if (metrics.length > 0) {
      this.yAxisLabels.push(key);
      this.podMetrics[key] = [];
      this.podMetrics[key].push({ name: key, series: metrics });
    }
  }
}
