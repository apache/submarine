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
import { ExperimentService } from '@submarine/services/experiment.service';

@Component({
  selector: 'submarine-model-home',
  templateUrl: './model-home.component.html',
  styleUrls: ['./model-home.component.scss'],
})
export class ModelHomeComponent implements OnInit {
  constructor(private experimentService: ExperimentService) {}

  modelCards: ModelInfo[];

  onDisplayModelCards = [];
  
  nameForFilter = "";
  listOfTagsOption: Array<{ label: string; value: string }> = [];
  listOfChosenTags = [];

  ngOnInit() {
    this.modelCards = [
      {
        'name': "Model One",
        'createTime': "2021-10-12",
        'updatedTime': "2021-10-13", 
        'tags': ["image", 'text'],
        'description': "first model",
      }, 
      {
        'name': "Model Two",
        'createTime': "2021-10-12",
        'updatedTime': "2021-10-13", 
        'tags': ["speech"],
        'description': "second model",
      },
      {
        'name': "Model Three",
        'createTime': "2021-10-12",
        'updatedTime': "2021-10-13", 
        'tags': ["speech"],
        'description': "The third model has very long description for description shorten testing, 1234567890 A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs",
      },
    ];
    
    this.onDisplayModelCards = this.modelCards.map(card => card);
    let tags = [];
    this.modelCards.map((card) => {
      Array.prototype.push.apply(tags, card.tags);
    });
    let tags_set = new Set(tags);
    tags = Array.from(tags_set);
    this.listOfTagsOption = tags.map((tag) => ({ "label": String(tag), "value": String(tag)}));
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
