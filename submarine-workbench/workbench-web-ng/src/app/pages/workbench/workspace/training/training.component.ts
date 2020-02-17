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
  selector: 'app-training',
  templateUrl: './training.component.html',
  styleUrls: ['./training.component.scss']
})
export class TrainingComponent implements OnInit {
  isSpinning = true;
  constructor() { }

  categories = [
    {name: "Category1", enable:false},
    {name: "Category2", enable:false},
    {name: "Category3", enable:false},
    {name: "Category4", enable:false},
    {name: "Category5", enable:false},
    {name: "Category6", enable:false},
    {name: "Category7", enable:false}
  ];
  ownProcess = false;
  tagValue = ['a10', 'c12', 'tag'];
  userSelectedValue = 'noLimit';
  ratingSelectedValue = 'noLimit'
  activeUsers = ["John", "Jason"];
  ratings = ["Execellent", "Good", "Moderate"];
  
  ngOnInit() {
    
  }

  performChange(){
    console.log('cool')
  }

}
