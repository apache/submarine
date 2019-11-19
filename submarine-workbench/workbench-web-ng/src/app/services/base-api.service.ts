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
import { environment } from '@submarine/environment';

class HttpError extends Error {
  code: number;
  url: string;
  method: string;
  params: object;

  constructor(message: string, code: number, url?: string, method?: string, params?: object) {
    super(message);

    this.code = code;
    this.url = url;
    this.method = method;
    this.params = params;

    if (!environment.production) {
      this.logError();
    }
  }

  logError() {
    console.group('Http error');
    console.log('error message: ', this.message);
    console.log('error code: ', this.code);
    console.log('-------------------------------');
    console.log('url: ', this.url);
    console.log('method: ', this.method);
    console.log('params: ', this.params);
  }
}

@Injectable({
  providedIn: 'root'
})
export class BaseApiService {
  baseApi: string;

  getPort() {
    let port = Number(location.port);
    if (!port) {
      port = 80;
      if (location.protocol === 'https:') {
        port = 443;
      }
    }
    return port;
  }

  getBase() {
    return `${location.protocol}//${location.hostname}:${this.getPort()}`;
  }

  getRestApiBase() {
    if (!this.baseApi) {
      this.baseApi = this.skipTrailingSlash(this.getBase()) + '/api';
    }

    return this.baseApi;
  }

  getRestApi(str: string): string {
    return `${this.getRestApiBase()}${str}`;
  }

  createRequestError(message: string, code: number, url?: string, method?: string, params?: any) {
    return new HttpError(message, code, url, method, params);
  }

  private skipTrailingSlash(path) {
    return path.replace(/\/$/, '');
  }
}
