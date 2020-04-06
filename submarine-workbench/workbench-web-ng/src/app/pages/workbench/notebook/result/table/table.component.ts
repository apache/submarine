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

@Component({
  selector: 'submarine-table',
  templateUrl: './table.component.html',
  styleUrls: ['./table.component.scss']
})
export class TableComponent implements OnInit {
  sortName: string | null = null;
  sortValue: string | null = null;
  listOfData: Array<{ week: string; value: number }> = [
    { week: 'Mon', value: 820 },
    { week: 'Tue', value: 932 },
    { week: 'Wed', value: 901 },
    { week: 'Thu', value: 934 },
    { week: 'Fri', value: 1290 },
    { week: 'Sat', value: 1330 },
    { week: 'Sun', value: 1320 }
  ];
  listOfDisplayData: Array<{ week: string; value: number }> = [...this.listOfData];

  constructor() {
  }

  ngOnInit(): void {
  }

  sort(sort: { key: string; value: string }): void {
    this.sortName = sort.key;
    this.sortValue = sort.value;

    const data = this.listOfData;

    /** sort data **/
    if (this.sortName && this.sortValue) {
      this.listOfDisplayData = data.sort((a, b) =>
        this.sortValue === 'ascend'
          ? a[this.sortName!] > b[this.sortName!]
          ? 1
          : -1
          : b[this.sortName!] > a[this.sortName!]
          ? 1
          : -1
      );
    } else {
      this.listOfDisplayData = data;
    }
  }
}
