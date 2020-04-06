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

import { AfterViewInit, Component, ElementRef, ViewChild } from '@angular/core';
import * as echarts from 'echarts';

@Component({
  selector: 'submarine-line-chart',
  templateUrl: './line-chart.component.html',
  styleUrls: ['./line-chart.component.scss']
})
export class LineChartComponent implements AfterViewInit {
  @ViewChild('chartElement') chartElement: ElementRef;
  chartInstance: any = null;

  chartData: Array<{ week: string, value: number }> = [{ week: 'Mon', value: 820 },
    { week: 'Tue', value: 932 },
    { week: 'Wed', value: 901 },
    { week: 'Thu', value: 934 },
    { week: 'Fri', value: 1290 },
    { week: 'Sat', value: 1330 },
    { week: 'Sun', value: 1320 }];

  constructor() {
  }

  ngAfterViewInit(): void {
    this.chartInstance = echarts.init(this.chartElement.nativeElement);

    this.chartInstance.setOption({
      xAxis: {
        type: 'category',
        data: this.chartData.map(item => item.week)
      },
      yAxis: {
        type: 'value'
      },
      series: [{
        type: 'line',
        data: this.chartData.map(item => item.value)
      }]
    });

    setTimeout(() => {
      this.chartInstance.resize();
    });
  }
}
