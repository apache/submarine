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
import { ModelInfo } from '@submarine/interfaces/model-info';
import { ModelService } from '@submarine/services/model.service';

@Component({
  selector: 'submarine-model-home',
  templateUrl: './model-home.component.html',
  styleUrls: ['./model-home.component.scss'],
})
export class ModelHomeComponent implements OnInit {
  constructor(private modelService: ModelService) {}

  isModelCardsLoading: boolean = true;
  modelCards: ModelInfo[];

  onDisplayModelCards = [];
  
  nameForFilter = "";
  listOfTagsOption: Array<{ label: string; value: string }> = [];
  listOfChosenTags = [];

  ngOnInit() {
    this.fetchModelCards();
    this.modelService.emitInfo(null);
  }
  
  fetchModelCards = () => {
    this.modelService.fetchModelList().subscribe((res) => {
      this.modelCards = res;
      this.onDisplayModelCards = this.modelCards.map(card => card);
      let tags = [];
      this.modelCards.map((card) => {
        Array.prototype.push.apply(tags, card.tags);
      });
      let tags_set = new Set(tags);
      tags = Array.from(tags_set);
      this.listOfTagsOption = tags.map((tag) => ({ "label": String(tag), "value": String(tag)}));
      this.isModelCardsLoading = false;
    });
  }

  searchModel(event: any) {
    this.nameForFilter = event.target.value;
    this.changeDisplayModelCards();
  }
  
  filterByTags() {
    this.changeDisplayModelCards();
  }

  changeDisplayModelCards() {
    this.onDisplayModelCards = this.modelCards.filter((card) => card.name.toLowerCase().includes(this.nameForFilter.toLowerCase()));
    this.onDisplayModelCards = this.onDisplayModelCards.filter((card) => {
      for (let chosenTag of this.listOfChosenTags) {
        if (!card.tags.includes(chosenTag)) return false;
      }
      return true;
    });
  }
}
