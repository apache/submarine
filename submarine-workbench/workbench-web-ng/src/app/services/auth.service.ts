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

import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Rest, SysUser } from '@submarine/interfaces';
import * as md5 from 'md5';
import { of, Observable } from 'rxjs';
import { map, switchMap } from 'rxjs/operators';
import { BaseApiService } from './base-api.service';
import { LocalStorageService } from './local-storage.service';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  isLoggedIn = false;
  authTokenKey = 'auth_token';

  // store the URL so we can redirect after logging in
  redirectUrl: string;

  constructor(
    private localStorageService: LocalStorageService,
    private baseApi: BaseApiService,
    private httpClient: HttpClient
  ) {
    const authToken = this.localStorageService.get<string>(this.authTokenKey);
    this.isLoggedIn = !!authToken;
  }

  login(userForm: { userName: string; password: string }): Observable<SysUser> {
    const apiUrl = this.baseApi.getRestApi('/auth/login');
    const params = {
      username: userForm.userName,
      password: md5(userForm.password)
    };

    return this.httpClient.post<Rest<SysUser>>(apiUrl, params).pipe(
      switchMap(res => {
        if (res.success) {
          this.isLoggedIn = true;
          this.localStorageService.set(this.authTokenKey, res.result.token);
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', params);
        }
      })
    );
  }

  logout() {
    return this.httpClient.post<Rest<boolean>>(this.baseApi.getRestApi('/auth/logout'), {}).pipe(
      map(res => {
        if (res.result) {
          this.isLoggedIn = false;
          this.localStorageService.remove(this.authTokenKey);
        }

        return res.result;
      })
    );
  }
}
