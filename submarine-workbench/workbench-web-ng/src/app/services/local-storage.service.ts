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

import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class LocalStorageService {
  prefix = 'submarine';

  generateKey(key: string) {
    return `${this.prefix}_${key.toLowerCase()}`;
  }

  get<T>(key: string): T {
    try {
      return JSON.parse(localStorage.getItem(this.generateKey(key)));
    } catch (e) {
      return null;
    }
  }

  set(key: string, value: string | number | object | any[]) {
    try {
      const saveValue = JSON.stringify(value);

      window.localStorage.setItem(this.generateKey(key), saveValue);
    } catch (e) {
      // empty
    }
  }

  remove(key: string) {
    window.localStorage.removeItem(this.generateKey(key));
  }
}
