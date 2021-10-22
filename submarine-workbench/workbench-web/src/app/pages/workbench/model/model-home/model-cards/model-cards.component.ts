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

import { Component, Input, OnInit } from '@angular/core';
import { ModelInfo } from '@submarine/interfaces/model-info';
@Component({
  selector: 'submarine-model-cards',
  templateUrl: './model-cards.component.html',
  styleUrls: ['./model-cards.component.scss'],
})
export class ModelCardsComponent implements OnInit {
  @Input() modelCards: ModelInfo[];
  nowPage: number;
  totalPages: number;
  onPageModelCards: ModelInfo[];
  pageUnit = 4;

  constructor() {}

  ngOnInit() {
    this.loadPage();
  }

  ngOnChanges() {
    this.loadPage();
  }

  loadPage() {
    this.nowPage = 1;
    this.totalPages = this.modelCards.length;
    this.loadOnPageModelCards(this.nowPage);
  }

  loadOnPageModelCards = (newPage: number) => {
    let start = this.pageUnit * (newPage - 1);
    this.onPageModelCards = this.modelCards.filter((_, index) => index < start + this.pageUnit && index >= start);
  }
}
