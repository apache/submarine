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

import { registerLocaleData } from '@angular/common';
import en from '@angular/common/locales/en';
import zh from '@angular/common/locales/zh';
import {en_US, zh_CN} from 'ng-zorro-antd';

const locales = [en, zh];
export const localeDict = {'en_US': en_US, 'zh_CN': zh_CN};

export function regLocales() {
  for (const locale of locales) {
    registerLocaleData(locale);
  }
}
