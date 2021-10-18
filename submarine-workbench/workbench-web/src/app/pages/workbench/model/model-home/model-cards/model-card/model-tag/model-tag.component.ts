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

@Component({
  selector: 'submarine-model-tag',
  templateUrl: './model-tag.component.html',
  styleUrls: ['./model-tag.component.scss'],
})
export class ModelTagComponent implements OnInit {
  @Input() tag: string;
  @Input() cssType: string;
  textColor: string;
  backgroundColor: string;
  borderColor: string;
  margin: string;
  closable: boolean;

  constructor() {}
  
  ngOnInit() {
    this.textColor = this.stringToColour(this.tag, "text");
    this.backgroundColor = this.stringToColour(this.tag, "background");
    this.borderColor = this.stringToColour(this.tag, "border");
    this.margin = this.cssType == "selection" ? "0px -20px 0px -10px" : "0 3px 0 3px";
    this.closable = this.cssType == "selection";
  }

  stringToColour = (str: string, type: string) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    let colour = '#';
    for (let i = 0; i < 3; i++) {
        let value = (hash >> (i * 8)) & 0xFF;
        colour += ('00' + value.toString(16)).substr(-2);
    }
    const hexToRgb = (hex: string) => {
      var bigint = parseInt(hex.substring(1,), 16);
      var r = (bigint >> 16) & 255;
      var g = (bigint >> 8) & 255;
      var b = bigint & 255;
  
      return r + "," + g + "," + b;
    }
    const opacity = type == "background" ? "0.1" : type == "border" ? "0.4" : "1.0";
    const rgb = "rgba(" + hexToRgb(colour) + "," + opacity + ")";
    return rgb;
  }
}
